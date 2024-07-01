from __future__ import annotations
import h5py
import numpy as np
import os
import polars as pl
import re
import sys
import warnings
from decimal import Decimal
from datetime import date, datetime, time, timedelta
from itertools import chain
from packaging import version
from pathlib import Path
from scipy.sparse import csr_array, csc_array, csr_matrix, csc_matrix, \
    hstack, vstack
from scipy.stats import mode, rankdata
from textwrap import fill
from typing import Any, Callable, ItemsView, Iterable, KeysView, Literal, \
    Mapping, Sequence, ValuesView
from utils import bonferroni, check_bounds, check_dtype, check_type, \
    cython_inline, cython_type, fdr, filter_columns, \
    generate_palette, is_integer, plural, prange, sparse_matrix_vector_op, \
    Timer, to_tuple
pl.enable_string_cache()

Color = str | float | np.floating | \
        tuple[int | np.integer, int | np.integer, int | np.integer] | \
        tuple[int | np.integer, int | np.integer, int | np.integer,
              int | np.integer] | \
        tuple[float | np.floating, float | np.floating,
              float | np.floating] | \
        tuple[float | np.floating, float | np.floating, float | np.floating,
              float | np.floating]
Indexer = int | np.integer | str | slice | \
          np.ndarray[Any, np.integer | np.bool_] | \
          pl.Series | list[int | np.integer | str | bool | np.bool_]
Scalar = str | int | float | Decimal | date | time | datetime | timedelta | \
         bool | bytes

def DE(self,
        label_column: str,
        covariate_columns: str | Iterable[str] | None,
        *,
        case_control: bool = True,
        include_library_size_as_covariate: bool = True,
        include_num_cells_as_covariate: bool = True,
        return_voom_info: bool = True,
        allow_float: bool = False,
        verbose: bool = True) -> DE:
    """
    Perform differential expression (DE) on a Pseudobulk dataset with
    limma-voom. Uses voomByGroup when case_control=True, which is better
    than regular voom for case-control DE.
    
    Loosely based on the `de_pseudobulk()` function from
    github.com/tluquez/utils/blob/main/utils.R, which is itself based on
    github.com/neurorestore/Libra/blob/main/R/pseudobulk_de.R.

    Args:
        label_column: the name of the column of obs to calculate DE with
                        respect to. If `case_control=True`, must be Boolean,
                        integer, floating-point, or Enum with cases = 1/True
                        and controls = 0/False. If `case_control=False`, must
                        be integer or floating-point.
        covariate_columns: the column(s) of obs to use as covariates, or
                            None to not include covariates
        case_control: whether the analysis is case-control or with respect
                        to a quantitative variable.
                        If True, uses voomByGroup instead of regular voom,
                        and uses obs[label_column] as the `group` argument
                        to calcNormFactors().
        include_library_size_as_covariate: whether to include the log2 of
                                            the library size, calculated
                                            according to the method of
                                            edgeR's calcNormFactors(), as an
                                            additional covariate
        include_num_cells_as_covariate: whether to include the log2 of the
                                        `'num_cells'` column of obs, i.e.
                                        the number of cells that went into
                                        each sample's pseudobulk in each
                                        cell type, as an additional
                                        covariate
        return_voom_info: whether to include the voom weights and voom plot
                            data in the returned DE object; set to False for
                            reduced runtime if you do not need to use the
                            voom weights or generate voom plots
        allow_float: if False, raise an error if `X.dtype` is
                        floating-point (suggesting the user may not be using
                        the raw counts, e.g. due to accidentally having run
                        log_CPM() already); if True, disable this sanity check
        verbose: whether to print out details of the DE estimation

    Returns:
        A DE object with a `table` attribute containing a polars DataFrame
        of the DE results. If `return_voom_info=True`, also includes a
        `voom_weights` attribute containing a {cell_type: DataFrame}
        dictionary of voom weights, and a `voom_plot_data` attribute
        containing a {cell_type: DataFrame} dictionary of info necessary to
        construct a voom plot with `DE.plot_voom()`.
    """
    # Import required Python and R packages
    from ryp import r, to_py, to_r
    r('suppressPackageStartupMessages(library(edgeR))')
    # Source voomByGroup code
    if case_control:
        r(self._voomByGroup_source_code)
    # Check inputs
    check_type(label_column, 'label_column', str, 'a string')
    if covariate_columns is not None:
        covariate_columns = to_tuple(covariate_columns)
        for column in covariate_columns:
            if not isinstance(column, str):
                error_message = (
                    f'All elements of `covariate_columns` must be '
                    f'strings, but `covariate_columns` contains an '
                    f'element of type `{type(column).__name__}`')
                raise TypeError(error_message)
    check_type(case_control, 'case_control', bool, 'boolean')
    check_type(include_library_size_as_covariate,
                'include_library_size_as_covariate', bool, 'boolean')
    check_type(include_num_cells_as_covariate,
                'include_num_cells_as_covariate', bool, 'boolean')
    if include_num_cells_as_covariate:
        for cell_type in self:
            if 'num_cells' not in self.obs[cell_type]:
                error_message = (
                    f'`include_num_cells_as_covariate` is True, but '
                    f'`num_cells` was not found in obs[{cell_type!r}]')
                raise KeyError(error_message)
        if covariate_columns is not None and \
                'num_cells' in covariate_columns:
            error_message = (
                f'`include_num_cells_as_covariate` is True, so '
                f'`num_cells` should be removed from `covariate_columns` '
                f'since its log is already being included as a covariate')
            raise ValueError(error_message)
    check_type(allow_float, 'allow_float', bool, 'boolean')
    check_type(verbose, 'verbose', bool, 'boolean')
    # Compute DE for each cell type
    DE_results = {}
    if return_voom_info:
        voom_weights = {}
        voom_plot_data = {}
        ebayes_data = {}
    for cell_type, (X, obs, var) in self.items():
        # If `allow_float=False`, raise an error if `X` is floating-point
        if not allow_float and np.issubdtype(X.dtype, np.floating):
            error_message = (
                f"X[{cell_type!r}].dtype is {X.dtype}, a floating-point "
                f"data type; if you are sure that all values are integers "
                f"(i.e. (X[{cell_type!r}].data == X[{cell_type!r}].data"
                f".astype(int)).all()`), then set allow_float=True (or "
                f"just cast `X` to an integer dtype). Alternately, did "
                f"you accidentally run log_CPM() before DE()?")
            raise TypeError(error_message)
        # Check that `label_column` and all the `covariate_columns` are in
        # obs, and (if case_control=True) that `label_column` has only two
        # unique values
        with Timer(f'[{cell_type}] Calculating DE', verbose=verbose):
            if np.issubdtype(X.dtype, np.floating):
                error_message = (
                    f'self.X[{cell_type!r}].dtype is {X.dtype}, a '
                    f'floating-point data type, but must be an integer '
                    f'data type; did you accidentally run log_CPM() '
                    f'before DE()?')
                raise ValueError(error_message)
            if label_column not in obs:
                error_message = (
                    f'label_column {label_column!r} is not a column of '
                    f'obs[{cell_type!r}]')
                raise KeyError(error_message)
            DE_labels = obs[label_column]
            check_dtype(DE_labels,
                        f'obs[{cell_type!r}][{label_column!r}]',
                        (pl.Boolean, 'integer', 'floating-point', pl.Enum)
                        if case_control else ('integer', 'floating-point'))
            DE_label_null_count = DE_labels.null_count()
            if DE_label_null_count > 0:
                error_message = (
                    f'obs[{cell_type!r}][{label_column!r}] contains '
                    f'{DE_label_null_count:,} '
                    f'{plural("null value", DE_label_null_count)}, but '
                    f'must not contain any')
                raise ValueError(error_message)
            if covariate_columns is not None:
                for column in covariate_columns:
                    if column not in obs:
                        error_message = (
                            f'`covariate_columns` contains the string '
                            f'{column!r}, which is not a column of '
                            f'obs[{cell_type!r}]')
                        raise KeyError(error_message)
                if include_num_cells_as_covariate:
                    covariates = obs.select(*covariate_columns,
                                            pl.col.num_cells.log(2))
                else:
                    covariates = obs.select(covariate_columns)
            else:
                if include_num_cells_as_covariate:
                    covariates = obs.select(pl.col.num_cells.log(2))
                else:
                    covariates = pl.DataFrame()
            for column, null_count in \
                    covariates.null_count().melt().rows():
                if null_count > 0:
                    error_message = (
                        f'obs[{cell_type!r}][{column!r}] contains '
                        f'{null_count:,} '
                        f'{plural("null value", null_count)}, but must '
                        f'not contain any')
                    raise ValueError(error_message)
            if case_control and DE_labels.dtype != pl.Boolean:
                if DE_labels.dtype == pl.Enum:
                    categories = DE_labels.cat.get_categories()
                    if len(categories) != 2:
                        error_message = (
                            f'obs[{cell_type!r}][{label_column!r}] is an '
                            f'Enum column with {len(categories):,} categor'
                            f'{"y" if len(categories) == 1 else "ies"}, '
                            f'but must have 2 (cases and controls)')
                        raise ValueError(error_message)
                    DE_labels = DE_labels.to_physical()
                else:
                    unique_labels = DE_labels.unique()
                    num_unique_labels = len(unique_labels)
                    if num_unique_labels != 2:
                        plural_string = \
                            plural('unique value', num_unique_labels)
                        error_message = (
                            f'obs[{cell_type!r}][{label_column}] is a '
                            f'numeric column with {num_unique_labels:,} '
                            f'{plural_string}, but must have 2 '
                            f'(cases = 1, controls = 0) unless '
                            f'case_control=False')
                        raise ValueError(error_message)
                    if not unique_labels.sort().equals(
                            pl.Series(label_column, [0, 1])):
                        error_message = (
                            f'obs[{cell_type!r}][{label_column!r}] is a '
                            f'numeric column with 2 unique values, '
                            f'{unique_labels[0]} and {unique_labels[1]}, '
                            f'but must have cases = 1 and controls = 0')
                        raise ValueError(error_message)
            # Get the design matrix
            if verbose:
                print('Generating design matrix...')
            design_matrix = \
                obs.select(pl.lit(1).alias('intercept'), DE_labels)
            if covariates.width:
                design_matrix = pl.concat([
                    design_matrix,
                    covariates.to_dummies(covariates.select(
                        pl.col(pl.Categorical, pl.Enum)).columns,
                        drop_first=True)],
                    how='horizontal')
            try:
                if include_library_size_as_covariate:
                    if verbose:
                        print('Estimating library size...')
                    library_size = \
                        X.sum(axis=1) * self._calc_norm_factors(X.T)
                    if library_size.min() == 0:
                        error_message = \
                            f'Some library sizes are 0 in {cell_type}'
                        raise ValueError(error_message)
                    to_r(library_size, '.Pseudobulk.library.size',
                            rownames=obs['ID'])
                    design_matrix = design_matrix\
                        .with_columns(library_size=np.log2(library_size))
                to_r(design_matrix, '.Pseudobulk.design.matrix',
                        rownames=obs['ID'])
                # Convert the expression matrix to R
                if verbose:
                    print('Converting the expression matrix to R...')
                to_r(X.T, '.Pseudobulk.X.T', rownames=var[:, 0],
                        colnames=obs['ID'])
                # Run voom
                to_r(return_voom_info, 'save.plot')
                if case_control:
                    if verbose:
                        print('Running voomByGroup...')
                    to_r(DE_labels, '.Pseudobulk.DE.labels',
                            rownames=obs['ID'])
                    r('.Pseudobulk.voom.result = voomByGroup('
                        '.Pseudobulk.X.T, .Pseudobulk.DE.labels, '
                        '.Pseudobulk.design.matrix, '
                        '.Pseudobulk.library.size, save.plot=save.plot, '
                        'print=FALSE)')
                else:
                    if verbose:
                        print('Running voom...')
                    r('.Pseudobulk.voom.result = voom(.Pseudobulk.X.T, '
                        '.Pseudobulk.design.matrix, '
                        '.Pseudobulk.library.size, save.plot=save.plot)')
                if return_voom_info:
                    # noinspection PyUnboundLocalVariable
                    voom_weights[cell_type] = \
                        to_py('.Pseudobulk.voom.result$weights',
                                index='gene')
                    # noinspection PyUnboundLocalVariable
                    voom_plot_data[cell_type] = pl.DataFrame({
                        f'{prop}_{dim}_{case}': to_py(
                            f'.Pseudobulk.voom.result$voom.{prop}$'
                            f'`{case_label}`${dim}', format='numpy')
                        for prop in ('xy', 'line') for dim in ('x', 'y')
                        for case, case_label in zip(
                            (False, True), ('FALSE', 'TRUE')
                            if DE_labels.dtype == pl.Boolean else (0, 1))}
                        if case_control else {
                        f'{prop}_{dim}': to_py(
                            f'.Pseudobulk.voom.result$voom.{prop}${dim}',
                            format='numpy')
                        for prop in ('xy', 'line') for dim in ('x', 'y')})
                # Run limma
                if verbose:
                    print('Running lmFit...')
                r('.Pseudobulk.lmFit.result = lmFit('
                    '.Pseudobulk.voom.result, .Pseudobulk.design.matrix)')
                if verbose:
                    print('Running eBayes...')
                r('.Pseudobulk.eBayes.result = eBayes('
                    '.Pseudobulk.lmFit.result, trend=FALSE, robust=FALSE)')
                if return_voom_info:
                    ebayes_data[cell_type] = \
                    to_py('.Pseudobulk.eBayes.result', index=None)
                # Get results table
                if verbose:
                    print('Running topTable...')
                to_r(label_column, 'label.column')
                r('.Pseudobulk.topTable.result = topTable('
                    '.Pseudobulk.eBayes.result, coef=label.column, '
                    'number=Inf, adjust.method="none", sort.by="P", '
                    'confint=TRUE)')
                if verbose:
                    print('Collating results...')
                DE_results[cell_type] = \
                    to_py('.Pseudobulk.topTable.result', index='gene')\
                    .select('gene',
                            logFC=pl.col.logFC,
                            SE=to_py('.Pseudobulk.eBayes.result$s2.post')
                                .sqrt() *
                                to_py('.Pseudobulk.eBayes.result$stdev.'
                                        'unscaled[,1]', index=False),
                            LCI=pl.col('CI.L'),
                            UCI=pl.col('CI.R'),
                            AveExpr=pl.col.AveExpr,
                            P=pl.col('P.Value'),
                            Bonferroni=bonferroni(pl.col('P.Value')),
                            FDR=fdr(pl.col('P.Value')))
            finally:
                r('rm(list = Filter(exists, c(".Pseudobulk.X.T", '
                    '".Pseudobulk.DE.labels", ".Pseudobulk.design.matrix", '
                    '".Pseudobulk.library.size", ".Pseudobulk.voom.result", '
                    '".Pseudobulk.lmFit.result", '
                    '".Pseudobulk.eBayes.result", '
                    '".Pseudobulk.topTable.result")))')
    # Concatenate across cell types
    table = pl.concat([
        cell_type_DE_results
        .select(pl.lit(cell_type).alias('cell_type'), pl.all())
        for cell_type, cell_type_DE_results in DE_results.items()])
    if return_voom_info:
        return DE(table, case_control, voom_weights, voom_plot_data, 
                  ebayes_data)
    else:
        return DE(table, case_control)
    
    

class DE:
    """
    Differential expression results returned by Pseudobulk.DE().
    """
    
    def __init__(self,
                 table: pl.DataFrame,
                 case_control: bool | None = None,
                 voom_weights: dict[str, pl.DataFrame] | None = None,
                 voom_plot_data: dict[str, pl.DataFrame] | None = None,
                 ebayes_data: dict[str, pl.DataFrame] | None = None) -> \
            None:
        """
        Initialize the DE object.
        
        Args:
            table: a polars DataFrame containing the DE results, with columns:
                   - cell_type: the cell type in which DE was tested
                   - gene: the gene for which DE was tested
                   - logFC: the log fold change of the gene, i.e. its effect
                            size
                   - SE: the standard error of the effect size
                   - LCI: the lower 95% confidence interval of the effect size
                   - UCI: the upper 95% confidence interval of the effect size
                   - AveExpr: the gene's average expression in this cell type,
                              in log CPM
                   - P: the DE p-value
                   - Bonferroni: the Bonferroni-corrected DE p-value
                   - FDR: the FDR q-value for the DE
                   Or, a directory containing a DE object saved with `save()`.
            case_control: whether the analysis is case-control or with respect
                          to a quantitative variable. Must be specified unless
                          `table` is a directory.
            voom_weights: an optional {cell_type: DataFrame} dictionary of voom
                         weights, where rows are genes and columns are samples.
                         The first column of each cell type's DataFrame,
                         'gene', contains the gene names.
            voom_plot_data: an optional {cell_type: DataFrame} dictionary of
                            info necessary to construct a voom plot with
                            `DE.plot_voom()`
        """
        if isinstance(table, pl.DataFrame):
            check_type(case_control, 'case_control', bool, 'boolean')
            if voom_weights is not None:
                if voom_plot_data is None:
                    error_message = (
                        '`voom_plot_data` must be specified when '
                        '`voom_weights` is specified')
                    raise ValueError(error_message)
                check_type(voom_weights, 'voom_weights', dict, 'a dictionary')
                if voom_weights.keys() != voom_plot_data.keys():
                    error_message = (
                        '`voom_weights` and `voom_plot_data` must have '
                        'matching keys (cell types)')
                    raise ValueError(error_message)
                for key in voom_weights:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of `voom_weights` and `voom_plot_data` '
                            f'must be strings (cell types), but they contain '
                            f'a key of type `{type(key).__name__}`')
                        raise TypeError(error_message)
            if voom_plot_data is not None:
                if voom_weights is None:
                    error_message = (
                        '`voom_weights` must be specified when '
                        '`voom_plot_data` is specified')
                    raise ValueError(error_message)
                check_type(voom_plot_data, 'voom_plot_data', dict,
                           'a dictionary')
        elif isinstance(table, (str, Path)):
            table = str(table)
            if not os.path.exists(table):
                error_message = f'DE object directory {table!r} does not exist'
                raise FileNotFoundError(error_message)
            cell_types = [line.rstrip('\n') for line in
                          open(f'{table}/cell_types.txt')]
            voom_weights = {cell_type: pl.read_parquet(
                os.path.join(table, f'{cell_type.replace("/", "-")}.'
                                    f'voom_weights.parquet'))
                for cell_type in cell_types}
            voom_plot_data = {cell_type: pl.read_parquet(
                os.path.join(table, f'{cell_type.replace("/", "-")}.'
                                    f'voom_plot_data.parquet'))
                for cell_type in cell_types}
            # noinspection PyUnresolvedReferences
            case_control = \
                next(iter(voom_plot_data.values())).columns[0].count('_') == 2
            table = pl.read_parquet(os.path.join(table, 'table.parquet'))
        else:
            error_message = (
                f'`table` must be a polars DataFrame or a directory (string '
                f'or pathlib.Path) containing a saved DE object, but has type '
                f'{type(table).__name__}')
            raise TypeError(error_message)
        self.table = table
        self.case_control = case_control
        self.voom_weights = voom_weights
        self.voom_plot_data = voom_plot_data
    
    def __repr__(self) -> str:
        """
        Get a string representation of this DE object.
        
        Returns:
            A string summarizing the object.
        """
        num_cell_types = self.table['cell_type'].n_unique()
        descr = (
            f'DE object with {len(self.table):,} '
            f'{"entries" if len(self.table) != 1 else "entry"} across '
            f'{num_cell_types:,} {plural("cell type", num_cell_types)}')
        return descr
    
    def __eq__(self, other: DE) -> bool:
        """
        Test for equality with another DE object.
        
        Args:
            other: the other DE object to test for equality with

        Returns:
            Whether the two DE objects are identical.
        """
        if not isinstance(other, DE):
            error_message = (
                f'The left-hand operand of `==` is a DE object, but '
                f'the right-hand operand has type `{type(other).__name__}`')
            raise TypeError(error_message)
        return self.table.equals(other.table) and \
            self.case_control == other.case_control and \
            (other.voom_weights is None if self.voom_weights is None else
             self.voom_weights.keys() == other.voom_weights.keys() and
             all(self.voom_weights[cell_type].equals(
                     other.voom_weights[cell_type]) and
                 self.voom_plot_data[cell_type].equals(
                     other.voom_plot_data[cell_type])
                 for cell_type in self.voom_weights))
    
    def save(self, directory: str | Path, overwrite: bool = False) -> None:
        """
        Save a DE object to `directory` (which must not exist unless
        `overwrite=True`, and will be created) with the table at table.parquet.
        
        If the DE object contains voom info (i.e. was created with
        `return_voom_info=True` in `Pseudobulk.DE()`, the default), also saves
        each cell type's voom weights and voom plot data to
        f'{cell_type}_voom_weights.parquet' and
        f'{cell_type}_voom_plot_data.parquet', as well as a text file,
        cell_types.txt, containing the cell types.
        
        Args:
            directory: the directory to save the DE object to
            overwrite: if False, raises an error if the directory exists; if
                       True, overwrites files inside it as necessary
        """
        check_type(directory, 'directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        if not overwrite and os.path.exists(directory):
            error_message = (
                f'`directory` {directory!r} already exists; set '
                f'overwrite=True to overwrite')
            raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        self.table.write_parquet(os.path.join(directory, 'table.parquet'))
        if self.voom_weights is not None:
            with open(os.path.join(directory, 'cell_types.txt'), 'w') as f:
                print('\n'.join(self.voom_weights), file=f)
            for cell_type in self.voom_weights:
                escaped_cell_type = cell_type.replace('/', '-')
                self.voom_weights[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_weights.parquet'))
                self.voom_plot_data[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_plot_data.parquet'))
    
    def get_hits(self,
                 significance_column: str = 'FDR',
                 threshold: int | float | np.integer | np.floating = 0.05,
                 num_top_hits: int | None = None):
        """
        Get all (or the top) differentially expressed genes by filtering the
        DataFrame returned by Pseudobulk.DE().
        
        Args:
            significance_column: the name of a Boolean column of self.table to
                                 determine significance from
            threshold: the significance threshold corresponding to
                       significance_column
            num_top_hits: the number of top hits to report for each cell type;
                          if None, report all hits

        Returns:
            DE_results, subset to the (top) DE hits.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = \
                '`significance_column` is not a column of self.table'
            raise KeyError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        if num_top_hits is not None:
            check_type(num_top_hits, 'num_top_hits', int, 'a positive integer')
            check_bounds(num_top_hits, 'num_top_hits', 1)
        return self.table\
            .filter(pl.col(significance_column) < threshold)\
            .pipe(lambda df: df.group_by('cell_type', maintain_order=True)
                  .head(num_top_hits) if num_top_hits is not None else df)
    
    def get_num_hits(self,
                     significance_column: str = 'FDR',
                     threshold: int | float | np.integer | np.floating = 0.05):
        """
        Get the number of differentially expressed genes in each cell type.
        
        Args:
            significance_column: the name of a Boolean column of self.table to
                                 determine significance from
            threshold: the significance threshold corresponding to
                       significance_column

        Returns:
            A DataFrame with one row per cell type and two columns:
            'cell_type' and 'num_hits'.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = \
                '`significance_column` is not a column of self.table'
            raise KeyError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        return self.table\
            .filter(pl.col(significance_column) < threshold)\
            .group_by('cell_type', maintain_order=True)\
            .agg(num_hits=pl.len())\
            .sort('cell_type')
    
    def plot_voom(self,
                  directory: str | Path,
                  *,
                  point_color: Color = '#666666',
                  case_point_color: Color = '#ff6666',
                  point_size: int | float | np.integer | np.floating = 1,
                  case_point_size: int | float | np.integer | np.floating = 1,
                  line_color: Color = '#000000',
                  case_line_color: Color = '#ff0000',
                  line_width: int | float | np.integer | np.floating = 1.5,
                  case_line_width: int | float | np.integer |
                                   np.floating = 1.5,
                  scatter_kwargs: dict[str, Any] | None = None,
                  case_scatter_kwargs: dict[str, Any] | None = None,
                  plot_kwargs: dict[str, Any] | None = None,
                  case_plot_kwargs: dict[str, Any] | None = None,
                  legend_labels: list[str] |
                                 tuple[str, str] = ('Controls', 'Cases'),
                  legend_kwargs: dict[str, Any] | None = None,
                  xlabel: str = 'Average log2(count + 0.5)',
                  xlabel_kwargs: dict[str, Any] | None = None,
                  ylabel: str = 'sqrt(standard deviation)',
                  ylabel_kwargs: dict[str, Any] | None = None,
                  title: bool | str | dict[str, str] |
                         Callable[[str], str] = False,
                  title_kwargs: dict[str, Any] | None = None,
                  despine: bool = True,
                  overwrite: bool = False,
                  PNG: bool = False,
                  savefig_kwargs: dict[str, Any] | None = None):
        """
        Generate a voom plot for each cell type that differential expression
        was calculated for.
        
        Voom plots consist of a scatter plot with one point per gene. They
        visualize how the mean expression of each gene across samples (x)
        relates to its variation across samples (y). The plot also includes a
        LOESS (also called LOWESS) fit, a type of non-linear curve fit, of the
        mean-variance (x-y) trend.
        
        Specifically, the x position of a gene's point is the average, across
        samples, of the base-2 logarithm of the gene's count in each sample
        (plus a pseudocount of 0.5): in other words, mean(log2(count + 0.5)).
        The y position is the square root of the standard deviation, across
        samples, of the gene's log counts per million after regressing out,
        across samples, the differential expression design matrix.
        
        For case-control differential expression (`case_control=True` in
        `Pseudobulk.DE()`), voom is run separately for cases and controls
        ("voomByGroup"), and so the voom plots will show a separate LOESS
        trendline for each of the two groups, with the points and trendlines
        for the two groups shown in different colors.
        
        Args:
            directory: the directory to save voom plots to; will be created if
                       it does not exist. Each cell type's voom plot will be
                       saved to f'{cell_type}.pdf' in this directory, or
                       f'{cell_type}.png' if `PNG=True`.
            point_color: the color of the points in the voom plot; if
                         case-control, only points for controls will be plotted
                         in this color
            case_point_color: the color of the points for cases; ignored for
                              non-case-control differential expression
            point_size: the size of the points in the voom plot; if
                        case-control, only the control points will be plotted
                        with this size
            case_point_size: the size of the points for cases; ignored for
                             non-case-control differential expression
            line_color: the color of the LOESS trendline in the voom plot; if
                        case-control, only the control trendline will be
                        plotted in this color
            case_line_color: the color of the LOESS trendline for cases;
                             ignored for non-case-control differential
                             expression
            line_width: the width of the LOESS trendline in the voom plot; if
                        case-control, only the control trendline will be
                        plotted with this width
            case_line_width: the width of the LOESS trendline for cases;
                             ignored for non-case-control differential
                             expression
            scatter_kwargs: a dictionary of keyword arguments to be passed to
                            `ax.scatter()`, such as:
                            - `rasterized`: whether to convert the scatter plot
                              points to a raster (bitmap) image when saving to
                              a vector format like PDF. Defaults to True,
                              instead of the Matplotlib default of False.
                            - `marker`: the shape to use for plotting each cell
                            - `norm`, `vmin`, and `vmax`: control how the
                              numbers in `color_column` are converted to
                              colors, if `color_column` is numeric
                            - `alpha`: the transparency of each point
                            - `linewidths` and `edgecolors`: the width and
                              color of the borders around each marker. These
                              are absent by default (`linewidths=0`), unlike
                              Matplotlib's default. Both arguments can be
                              either single values or sequences.
                            - `zorder`: the order in which the cells are
                              plotted, with higher values appearing on top of
                              lower ones.
                            Specifying `s` or `c`/`color` will raise an error,
                            since these arguments conflict with the
                            `point_size` and `point_color` arguments,
                            respectively.
                            If case-control and `case_scatter_kwargs` is not
                            None, these settings only apply to control points.
            case_scatter_kwargs: a dictionary of keyword arguments to be passed
                                 to `plt.scatter()` for case points. Like for 
                                 `scatter_kwargs`, `rasterized=True` is the 
                                 default, and specifying `s` or `c`/`color`
                                 will raise an error. If None and
                                 `scatter_kwargs` is not None, the settings in
                                 `scatter_kwargs` apply to all points. Can only
                                 be specified for case-control differential
                                 expression.
            plot_kwargs: a dictionary of keyword arguments to be passed to
                         `plt.plot()` when plotting the trendlines, such as
                         `linestyle` for dashed trendlines. Specifying
                         `color`/`c` or `linewidth` will raise an error, since
                         these arguments conflict with the `line_color` and
                         `line_width` arguments, respectively.
            case_plot_kwargs: a dictionary of keyword arguments to be passed to
                              `plt.plot()` when plotting the case trendlines.
                              Specifying `color`/`c` or `linewidth` will raise
                              an error, like for `plot_kwargs`. If None and
                              `plot_kwargs` is not None, the settings in
                              `plot_kwargs` apply to all points. Can only be
                              specified for case-control differential
                              expression.
            legend_labels: a two-element tuple or list of labels for controls
                           and cases (in that order) in the legend, or None to
                           not include a legend. Ignored for non-case-control
                           differential expression.
            legend_kwargs: a dictionary of keyword arguments to be passed to
                           `plt.legend()` to modify the legend, such as:
                           - `loc`, `bbox_to_anchor`, and `bbox_transform` to
                             set its location.
                           - `prop`, `fontsize`, and `labelcolor` to set its
                             font properties
                           - `facecolor` and `framealpha` to set its background
                             color and transparency
                           - `frameon=True` or `edgecolor` to add or color
                             its border (`frameon` is False by default,
                             unlike Matplotlib's default of True)
                           - `title` to add a legend title
                           Can only be specified for case-control differential
                           expression.
            xlabel: the x-axis label for each voom plot, or None to not include
                    an x-axis label
            xlabel_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.xlabel()` to control the text properties, such
                          as `color` and `size` to modify the text color/size
            ylabel: the y-axis label for each voom plot, or None to not include
                    a y-axis label
            ylabel_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.ylabel()` to control the text properties, such
                          as `color` and `size` to modify the text color/size
            title: what to use as the title. If False, do not include a
                   title. If True, use the cell type as a title. If a string,
                   use the string as the title for every cell type. If a
                   dictionary, use `title[cell_type]` as the title; every cell
                   type must be present in the dictionary. If a function or
                   other Callable, use `title(cell_type)` as the title.
            title_kwargs: a dictionary of keyword arguments to be passed to
                          `plt.title()` to control the text properties, such
                          as `color` and `size` to modify the text color/size.
                          Cannot be specified when `title=False`.
            despine: whether to remove the top and right spines (borders of the
                     plot area) from the voom plots
            overwrite: if False, raises an error if the directory exists; if
                       True, overwrites files inside it as necessary
            PNG: whether to save the voom plots in PNG instead of PDF format
            savefig_kwargs: a dictionary of keyword arguments to be passed to
                            `plt.savefig()`, such as:
                            - `dpi`: defaults to 300 instead of Matplotlib's
                              default of 150
                            - `bbox_inches`: the bounding box of the portion of
                              the figure to save; defaults to 'tight' (crop out
                              any blank borders) instead of Matplotlib's
                              default of None (save the entire figure)
                            - `pad_inches`: the number of inches of padding to
                              add on each of the four sides of the figure when
                              saving. Defaults to 'layout' (use the padding
                              from the constrained layout engine) instead of
                              Matplotlib's default of 0.1.
                            - `transparent`: whether to save with a transparent
                              background; defaults to True if saving to a PDF
                              (i.e. when `PNG=False`) and False if saving to
                              a PNG, instead of Matplotlib's default of always
                              being False.
        """
        import matplotlib.pyplot as plt
        # Make sure this DE object contains `voom_plot_data`
        if self.voom_plot_data is None:
            error_message = (
                'this DE object does not contain the `voom_plot_data` '
                'attribute, which is necessary to generate voom plots; re-run '
                'Pseudobulk.DE() with return_voom_info=True to include this '
                'attribute')
            raise AttributeError(error_message)
        # Check that `directory` is a string or pathlib.Path
        check_type(directory, 'voom_plot_directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        # Check that each of the colors are valid Matplotlib colors, and
        # convert them to hex
        for color, color_name in ((point_color, 'point_color'),
                                  (line_color, 'line_color'),
                                  (case_point_color, 'case_point_color'),
                                  (case_line_color, 'case_line_color')):
            if not plt.matplotlib.colors.is_color_like(color):
                error_message = \
                    f'`{color_name}` is not a valid Matplotlib color'
                raise ValueError(error_message)
        point_color = plt.matplotlib.colors.to_hex(point_color)
        line_color = plt.matplotlib.colors.to_hex(line_color)
        case_point_color = plt.matplotlib.colors.to_hex(case_point_color)
        case_line_color = plt.matplotlib.colors.to_hex(case_line_color)
        # Check that point sizes are positive numbers
        check_type(point_size, 'point_size', (int, float), 'a positive number')
        check_bounds(point_size, 'point_size', 0, left_open=True)
        check_type(case_point_size, 'case_point_size', (int, float),
                   'a positive number')
        check_bounds(case_point_size, 'case_point_size', 0, left_open=True)
        # For each of the kwargs arguments, if the argument is not None, check
        # that it is a dictionary and that all its keys are strings.
        for kwargs, kwargs_name in (
                (scatter_kwargs, 'scatter_kwargs'),
                (case_scatter_kwargs, 'case_scatter_kwargs'),
                (plot_kwargs, 'plot_kwargs'),
                (case_plot_kwargs, 'case_plot_kwargs'),
                (legend_kwargs, 'legend_kwargs'),
                (xlabel_kwargs, 'xlabel_kwargs'),
                (ylabel_kwargs, 'ylabel_kwargs'),
                (title_kwargs, 'title_kwargs')):
            if kwargs is not None:
                check_type(kwargs, kwargs_name, dict, 'a dictionary')
                for key in kwargs:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of `{kwargs_name}` must be strings, '
                            f'but it contains a key of type '
                            f'`{type(key).__name__}`')
                        raise TypeError(error_message)
        # Check that `case_scatter_kwargs` and `case_plot_kwargs` are None for
        # non-case-control differential expression. If None, use the settings
        # from `scatter_kwargs` and `plot_kwargs`, respectively. Also set
        # `plot_kwargs` to {} if it is None.
        if case_scatter_kwargs is None:
            case_scatter_kwargs = scatter_kwargs
        elif not self.case_control:
            error_message = (
                '`case_scatter_kwargs` can only be specified for '
                'case-control differential expression')
            raise ValueError(error_message)
        if plot_kwargs is None:
            plot_kwargs = {}
        if case_plot_kwargs is None:
            case_plot_kwargs = plot_kwargs
        elif not self.case_control:
            error_message = (
                '`case_plot_kwargs` can only be specified for case-control '
                'differential expression')
            raise ValueError(error_message)
        # Override the defaults for certain keys of `scatter_kwargs` and
        # `case_scatter_kwargs`
        default_scatter_kwargs = dict(rasterized=True, linewidths=0)
        scatter_kwargs = default_scatter_kwargs | scatter_kwargs \
            if scatter_kwargs is not None else default_scatter_kwargs
        case_scatter_kwargs = default_scatter_kwargs | case_scatter_kwargs \
            if case_scatter_kwargs is not None else default_scatter_kwargs
        # Check that `scatter_kwargs` and `case_scatter_kwargs` do not contain
        # the `s` or `c`/`color` keys, and that `plot_kwargs` and
        # `case_plot_kwargs` do not contain the `c`/`color` or `linewidth` keys
        for plot, kwargs_set in enumerate(
                (((scatter_kwargs, 'scatter_kwargs'),
                  (case_scatter_kwargs, 'case_scatter_kwargs')),
                 ((plot_kwargs, 'plot_kwargs'),
                  (case_plot_kwargs, 'case_plot_kwargs')))):
            for kwargs, kwargs_name in kwargs_set:
                if kwargs is None:
                    continue
                for bad_key, alternate_argument in (
                        ('linewidth', 'line_width') if plot else
                        ('s', 'point_size'),
                        ('c', 'point_color' if plot else 'line_color'),
                        ('color', 'point_color' if plot else 'line_color')):
                    if bad_key in kwargs:
                        error_message = (
                            f"'{bad_key}' cannot be specified as a key in "
                            f"`{kwargs_name}`; specify the "
                            f"`{alternate_argument}` argument instead")
                        raise ValueError(error_message)
        # Check that `legend_labels` is a two-element tuple or list of strings
        check_type(legend_labels, 'legend_labels', (tuple, list),
                   'a length-2 tuple or list of strings')
        if len(legend_labels) != 2:
            error_message = (
                f'`legend_labels` must have a length of 2, but has a length '
                f'of {len(legend_labels):,}')
            raise ValueError(error_message)
        check_type(legend_labels[0], 'legend_labels[0]', str, 'a string')
        check_type(legend_labels[1], 'legend_labels[1]', str, 'a string')
        # Override the defaults for certain values of `legend_kwargs`; check
        # that it is None for non-case-control differential expression
        default_legend_kwargs = dict(frameon=False)
        if legend_kwargs is not None:
            if not self.case_control:
                error_message = (
                    '`legend_kwargs` can only be specified for case-control '
                    'differential expression')
                raise ValueError(error_message)
            legend_kwargs = default_legend_kwargs | legend_kwargs
        else:
            legend_kwargs = default_legend_kwargs
        # Check that `xlabel` and `ylabel` are strings, or None
        if xlabel is not None:
            check_type(xlabel, 'xlabel', str, 'a string')
        if ylabel is not None:
            check_type(ylabel, 'ylabel', str, 'a string')
        # Check that `title` is boolean, a string, a dictionary where all keys
        # are cell types and every cell type is present, or a Callable
        check_type(title, 'title', (bool, str, dict, Callable),
                   'boolean, a string, a dictionary, or a Callable')
        if isinstance(title, dict):
            if len(title) != len(self.voom_plot_data) or \
                    set(title) != set(self.voom_plot_data):
                error_message = (
                    'when `title` is a dictionary, all its keys must be cell '
                    'types, and every cell type must be present')
                raise ValueError(error_message)
        # Check that `title_kwargs` is None when `title=False`
        if title is False and title_kwargs is not None:
            error_message = \
                '`title_kwargs` cannot be specified when `title=False`'
            raise ValueError(error_message)
        # Check that `overwrite` and `PNG` are boolean
        check_type(overwrite, 'overwrite', bool, 'boolean')
        check_type(PNG, 'PNG', bool, 'boolean')
        # Override the defaults for certain values of `savefig_kwargs`
        default_savefig_kwargs = \
            dict(dpi=300, bbox_inches='tight', pad_inches='layout',
                 transparent=not PNG)
        savefig_kwargs = default_savefig_kwargs | savefig_kwargs \
            if savefig_kwargs is not None else default_savefig_kwargs
        # Create plot directory
        if not overwrite and os.path.exists(directory):
            error_message = (
                f'`directory` path {directory!r} already exists; set '
                f'overwrite=True to overwrite')
            raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        # Save each cell type's voom plot in this directory
        add_legend = self.case_control and legend_labels is not None
        for cell_type, voom_plot_data in self.voom_plot_data.items():
            voom_plot_file = os.path.join(
                directory,
                f'{cell_type.replace("/", "-")}.{"png" if PNG else "pdf"}')
            if self.case_control:
                if add_legend:
                    legend_patches = []
                for case in False, True:
                    plt.scatter(voom_plot_data[f'xy_x_{case}'],
                                voom_plot_data[f'xy_y_{case}'],
                                s=case_point_size if case else point_size,
                                c=case_point_color if case else point_color,
                                **(case_scatter_kwargs if case else
                                   scatter_kwargs))
                    plt.plot(voom_plot_data[f'line_x_{case}'],
                             voom_plot_data[f'line_y_{case}'],
                             c=case_line_color if case else line_color,
                             linewidth=case_line_width if case else line_width,
                             **(case_plot_kwargs if case else plot_kwargs))
                    if add_legend:
                        # noinspection PyUnboundLocalVariable
                        # noinspection PyUnresolvedReferences
                        legend_patches.append(
                            plt.matplotlib.patches.Patch(
                                facecolor=case_point_color if case else
                                point_color,
                                edgecolor=case_line_color if case else
                                line_color,
                                linewidth=case_line_width if case else
                                line_width,
                                label=legend_labels[case]))
                if add_legend:
                    plt.legend(handles=legend_patches, **legend_kwargs)
            else:
                plt.scatter(voom_plot_data['xy_x'], voom_plot_data['xy_y'],
                            s=point_size, c=point_color, **scatter_kwargs)
                plt.plot(voom_plot_data['line_x'], voom_plot_data['line_y'],
                         c=line_color, linewidth=line_width, **plot_kwargs)
            if xlabel_kwargs is None:
                xlabel_kwargs = {}
            if ylabel_kwargs is None:
                ylabel_kwargs = {}
            plt.xlabel(xlabel, **xlabel_kwargs)
            plt.ylabel(ylabel, **ylabel_kwargs)
            if title is not False:
                if title_kwargs is None:
                    title_kwargs = {}
                # noinspection PyCallingNonCallable
                plt.title(title[cell_type] if isinstance(title, dict)
                          else title if isinstance(title, str) else
                          title(cell_type) if isinstance(title, Callable) else
                          cell_type, **title_kwargs)
            if despine:
                spines = plt.gca().spines
                spines['top'].set_visible(False)
                spines['right'].set_visible(False)
            plt.savefig(voom_plot_file, **savefig_kwargs)
            plt.close()
