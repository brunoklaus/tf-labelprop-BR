'''
Created on 31 de dez de 2020

@author: klaus
'''
import os
import re

import numpy as np
import os.path as osp
import pandas as pd
from tf_labelprop.experiment.prefixes import AFFMAT_PREFIX, NOISE_PREFIX, ALG_PREFIX, FILTER_PREFIX, GENERAL_PREFIX, OUTPUT_PREFIX, INPUT_PREFIX
from tf_labelprop.logging.logger import log as LOG
from tf_labelprop.output.folders import CSV_FOLDER, PLOT_FOLDER, RESULTS_FOLDER, \
    VIS_FOLDER


def fix_underscore(x):
    return str(x).replace('_','\\textunderscore ')


def default_translate():
    dct = {
        f'{INPUT_PREFIX}dataset': 'Dataset',
        f'{INPUT_PREFIX}labeled_percent': '\\% labels',
        f'{INPUT_PREFIX}num_labeled': 'labels',
        f'{INPUT_PREFIX}benchmark': 'Benchmark',
        
        
        f'{AFFMAT_PREFIX}dist_func': 'Affmat dist.',
        f'{AFFMAT_PREFIX}sigma': '$\\sigma$',
        f'{AFFMAT_PREFIX}k': 'k',
        f'{AFFMAT_PREFIX}row_normalize':'Normalize rows',
        
        
        f'{FILTER_PREFIX}mu':'$\\mu$',
        f'{FILTER_PREFIX}filter':'Filter',
        
        
        f'{NOISE_PREFIX}corruption_level':'Noise \\%',
        f'{NOISE_PREFIX}deterministic':'Deterministic noise',
        f'{NOISE_PREFIX}type':'Noise type',
        
        f'{ALG_PREFIX}algorithm':'Classifier',
        f'{ALG_PREFIX}alpha':'$\\alpha$',
        f'{ALG_PREFIX}mu':'$\\mu$',
        f'{ALG_PREFIX}optimize_labels':'opt.l.',
        f'{ALG_PREFIX}custom_conv':'c.conv.',
        
        
        
        f'{OUTPUT_PREFIX}acc':'Acc.',
        f'{OUTPUT_PREFIX}acc_labeled':'L.Acc.',
        f'{OUTPUT_PREFIX}acc_unlabeled':'Ul.Acc.',
        
        f'{OUTPUT_PREFIX}CMN_acc':'Acc. (w/ CMN)',
        f'{OUTPUT_PREFIX}CMN_acc':'Acc. (labeled, w/ CMN)',
        f'{OUTPUT_PREFIX}CMN_acc':'Acc. (unlabeled, w/ CMN)',

        f'{OUTPUT_PREFIX}affmat_time':'Time (affmat)',
        f'{OUTPUT_PREFIX}classifier_time':'Time (classifier)',
        f'{OUTPUT_PREFIX}filter_time':'Time (classifier)',
        f'{OUTPUT_PREFIX}noise_time':'Time (noise)'
        }
    dct2 = dict(dct)
    for k in dct.keys():
        if k.startswith(OUTPUT_PREFIX):
            for x,y in zip(['mean','median','min','max','sd'],
                           ['','median','min','max','sd']):
                if len(y) > 0:
                    dct2[k+f'_{x}'] = dct[k] + f'({y})'
                else:
                    dct2[k+f'_{x}'] = dct[k]
    return dct2



def tex_table(df,y_vars,t_vars,vdiv_vars,hdiv_vars,c_vars,use_std,OUT_FOLDER,dct_translate = default_translate(),fixed_vars=[],bold_max=True):
    """
        Creates and saves a table in LaTeX corresponding to the observation of aggregated variables.
        
        
            Args:
                df (pd.DataFrame): Dataframe containing aggregated observations
                y_vars (List[str]): List of output values to be present.
                t_vars (List[str]): Variables used to define tables. There'll be one table corresponding to each combination of these variables.
                vdiv_vars (List[str]):  List of variables that correspond to vertical divisions. 
                hdiv_vars (List[str]): Variables used to represent horizontal rules. In each table, there'll be a horizontal rule separating each combination of these variables.
                c_vars (List[str]): Variables used to represent configurations. In each horizontal rule defined by `hdiv_vars`, there'll be one line for each combination of `c_vars` variables.
                use_sd (bool): If ``True``, will show +- 1 standard deviation from the mean as a shaded region.
                OUT_FOLDER (str): Folder to which the visualization will be saved.
                dct_translate (Dict[str,str]): Maps the columns of the DataFrame to strings that are more readable.
                fixed_vars (List[str]): List of fixed variables
                bold_max (bool): If ``True``, automatically makes the text of the largest value 
                
    """
    
    if len(t_vars) > 0:
        for g, g_df in df.groupby(t_vars):
            if len(t_vars) == 1:
                g = [g]
            else:
                g = list(g)
            
            _path = [OUT_FOLDER] + [x+'='+y for x,y in sorted(zip(t_vars,g))]
            _path = osp.join(*_path)
            tex_table(g_df,y_vars,[],vdiv_vars,hdiv_vars,c_vars,use_std,
                      OUT_FOLDER=_path,
                      dct_translate=dct_translate,fixed_vars=t_vars+fixed_vars,bold_max=bold_max)
    else:
        """
            Base case
        """
        fixed_vars_desc =  ';\n'.join([c + ":   "+  ';'.join(pd.unique(df[c]).astype(np.str)) for c in fixed_vars ])
        _tex = []
        
        """
            Title
        """
        print(OUT_FOLDER)
        if not osp.isdir(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)
        with open(osp.join(OUT_FOLDER,'table.txt'), "w") as _tex:
            _tex.write("%\\usepackage{tabularx}\n")
            _tex.write("%\\usepackage{booktabs}\n")
            _tex.write("\\begin{table}[h] \\tiny")
            
            df[:] = df[:].fillna('---')
            for c in hdiv_vars+c_vars+vdiv_vars:
                if df[c].dtype.kind in 'biufc':
                    df[c] = df[c].fillna(-1)
                else:
                    df[c] = df[c].fillna('NA').astype('string')
                df[c] = pd.Categorical(df[c]).as_ordered()
            df[:] = df.sort_values(by=hdiv_vars+c_vars+vdiv_vars)
            
            
            """
                -------------------
                    Create combination of all vdiv_vars with y_vars
                ----------------
            """
            lvls= [list(np.sort(pd.unique(df[y]))) for y in vdiv_vars]
            lvls.append(y_vars)
            import itertools
            comb_v = list(itertools.product(*lvls))
            comb_v = [list(zip(vdiv_vars + [OUTPUT_PREFIX],x)) for x in comb_v]
            comb_v_dct = [dict(x) for x in comb_v]
            
            _tex.write("\\begin{tabular}{@{}|" +\
                        f'{"c"* len(c_vars+hdiv_vars)}' + 
                        f"{'|c'* len(comb_v)}" + "|@{}} " + "\cline{" +\
                        f"{len(c_vars+hdiv_vars)+1}-{len(c_vars+hdiv_vars) + len(comb_v)}" + "} \n")
            
            """ Create Table HEADER """
            rep = 1
            multicolumn_len = len(comb_v)
            for v in vdiv_vars:
                L = ["\multicolumn{" + str(len(c_vars+hdiv_vars))+"}{c|}{}"]
                unique = pd.unique(df[v])
                multicolumn_len = multicolumn_len // (len(unique))
                print(v)
                print(multicolumn_len)
                for _ in range(rep):
                    for u in unique:
                        L.append('\\multicolumn{'+str(multicolumn_len) + "}{c|}{"+ f"{dct_translate.get(v,v)}={u}"+"}")
                _tex.write(fix_underscore(' & '.join(L)))
                _tex.write('\\\\ \n')
                rep *= len(unique)
            #_tex.write(fix_underscore('&'.join(c_vars+hdiv_vars+comb_v_cols)))
            _tex.write("\\hline \n")
            L = [dct_translate.get(c,c) for c in hdiv_vars+c_vars] + ([dct_translate.get(y,y) for y in pd.unique(y_vars)]\
                                                                 * ( len(comb_v)//pd.unique(y_vars).shape[0]    ))
            _tex.write(fix_underscore(' & '.join(L)))
            _tex.write("\\\\ \\hline  \n")
            def groupby(df,vrs):
                if len(vrs) == 1:
                    for v, v_df in df.groupby(vrs):
                        yield [v], v_df
                elif len(vrs) == 0:
                    yield [], df
                else:
                    for v, v_df in df.groupby(vrs):
                        yield v, v_df
            
            df = df.sort_values(hdiv_vars+c_vars+vdiv_vars)
            N_row = df.shape[0]
            row_counter = 0
            for h, h_df in groupby(df,hdiv_vars):
                oldh_df = h_df
                
                def _apply(h_df,out_var):
                    maxval = np.argmax(h_df[out_var].values,axis=0)
                    y = out_var
                    h_df[out_var] = [str(x) for x in np.round(100*h_df[out_var],2)]
                    if y.endswith('_mean'):
                        old_y = y
                        y = y[:-len('_mean')]
                        y = y+'_sd'
                        #print(list(zip(h_df[old_y].values,h_df.loc[:,y].values)))
                        def f(x):
                            return "$\\mathbf{"+ x[0] + " \\pm " + str(np.round(100*x[1],2)) + "}$"
                        def g(x):
                            return "$" + x[0] + " \\pm " + str(np.round(100*x[1],2)) + "$" 
                        h_df[old_y] = [f(x) if j == maxval   else g(x)  for j,x in enumerate(zip(h_df[old_y].values,h_df[y].values))]
                        
                            
                    else:
                        h_df[y] = ["$\\mathbf{"+ x +"}$" if j == maxval   else '$'+x+'$' for j,x in enumerate(h_df[y].values)]
                    return h_df
                for y in y_vars:
                    from functools import partial
                    h_df[y] = h_df.groupby(vdiv_vars).apply(partial(_apply,out_var=y))[y]
                     
                for c, c_df in groupby(h_df,c_vars):
                    if c_df.shape[0] > 0:
                        row = [str(y)  for y in h] + [str(y)  for y in c]
                        #print("==========================")
                        for v, v_df in groupby(c_df,vdiv_vars):
                            print(v_df[y])
                            _r = [str(v_df[y].values[0])  if v_df[y].shape[0] > 0 else "---" for y in y_vars ]
                            row.extend(_r)
                        _tex.write(' & '.join(row))
                        #print(row)
                        row_counter += c_df.shape[0]
                        if row_counter < N_row:
                            _tex.write("\\\\ \n")
                        
                if row_counter < N_row:
                    _tex.write("\\hline \n")
                
            """
                FINALIZE Table
            """
            _tex.write("\\end{tabular}\n ")
            
            _tex.write("\n \\caption{" + fix_underscore(fixed_vars_desc) +  "}\n")
            _tex.write("\\end{table}")
            
                        
            
            
        
            
        

def line_plot(df,x_var,y_var,agg_var,p_vars,l_vars, use_std,OUT_FOLDER,dct_translate = default_translate(),fixed_var_desc=''):
    """
        Creates and saves a lineplot.
            Args:
                df (pd.DataFrame): Dataframe containing aggregated observations
                x_var (str): Variable to be plotted on x axis.
                y_var (str): Variable to be plotted on y axis.
                agg_var (str): Which aggregation to use. One of ``mean``,``median``,``min``,``max``.
                p_vars (List[str]): Variables used to define plots. There'll be one plot corresponding to each combination of these variables.
                l_vars (List[str]): Variables used to represent levels. When the variables in `p_vars` are fixed, there'll be one line for each combination of `l_vars` variables.
                use_sd (bool): If ``True``, will show +- 1 standard deviation from the mean as a shaded region.
                OUT_FOLDER (str): Folder to which the visualization will be saved.
                dct_translate (Dict[str,str]): Maps the columns of the DataFrame to strings that are more readable.
                fixed_var_desc (str): A string containing the description of the fixed variables of the dataframe. Will be placed at the bottom of the plot.
                
    """
    
    if len(p_vars) > 0:
        for g, g_df in df.groupby(p_vars):
            if len(p_vars) == 1:
                g = [g]
            else:
                g = list(g)
            
            
            _path = [OUT_FOLDER] + [x+'='+y for x,y in sorted(zip(p_vars,g))]
            _path = osp.join(*_path)
            line_plot(g_df,x_var,y_var,agg_var,[],l_vars,use_std,OUT_FOLDER=_path,dct_translate=dct_translate,fixed_var_desc=fixed_var_desc)
    else:
        """
            Base case
        """
        import matplotlib.pyplot as plt
        import matplotlib
        #matplotlib.use('TkAgg')
        
        if len(l_vars) == 0:
            df[' '] = ' '
            l_vars = [' ']
        
        """"
        ---------------------------------------
            DEFINE FONT SIZE
        --------------------------------
        
        """
        SMALL_SIZE = 14
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 22
        BIGGEST_SIZE = 28
        
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGEST_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rcParams.update({
            "text.usetex": True,
        })
        
        import re
        number_newlines = len(re.findall('\\n',fixed_var_desc))
        fig, ax = plt.subplots(1,figsize=(20,10+number_newlines//2))
        df[l_vars] = df[l_vars].fillna('---')
        for l, l_df in df.groupby(l_vars):
            if len(l_vars) == 1:
                l = [l]
            else:
                l = list(l)
            
            if len(l_df[x_var].dropna()) == 0:
                l_df[x_var] = '-'
            elif isinstance(l_df[x_var].dropna().values[0],str):
                l_df[x_var] = l_df[x_var].fillna('-')
            
            l_inst = ';'.join([dct_translate[x]+'='+fix_underscore(y) for x,y in sorted(zip(l_vars,l))])
            print(l_df[x_var].values)
            print(l_df[f'{y_var}_{agg_var}'].values)
            print(fix_underscore(l_inst))
            p = ax.plot(l_df[x_var].values,l_df[f'{y_var}_{agg_var}'].values,marker='o',
                    label=fix_underscore(l_inst),
                     linewidth=4, markersize=12)
            facecolor = matplotlib.colors.to_rgba(p[0].get_color(),alpha=0.1)
            edgecolor = matplotlib.colors.to_rgba(p[0].get_color(),alpha=0.8)
            

            if use_std:
                ax.fill_between(l_df[x_var],
                                l_df[f'{y_var}_mean'] - l_df[f'{y_var}_sd'],
                                l_df[f'{y_var}_mean'] + l_df[f'{y_var}_sd'],
                                facecolor = facecolor,
                                edgecolor= edgecolor
                                )
            
            
           
        print(OUT_FOLDER)
        plot_subtitle = fix_underscore(fixed_var_desc)
        ax.set_xlabel(fix_underscore(dct_translate.get(x_var,x_var)) +\
                      '\n\n' + plot_subtitle)
        ax.set_ylabel(fix_underscore(dct_translate.get(f'{y_var}_{agg_var}',f'{y_var}_{agg_var}')) )
        fig.tight_layout()
        plt.grid()
        
        plt.legend()
        if not osp.isdir(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)
        fig.savefig(osp.join(OUT_FOLDER,f'PLOT_{y_var}__from__{x_var}.png'))
        
def interactive_line_plot(df,OUT_FOLDER):
    
    print("Dataframe columns:")
    print('\n'.join(list(df.columns)))
    
    _y_var = None
    while _y_var is None:
        try:
            _y_var = input(f"Type dependent output variable (e.g. {OUTPUT_PREFIX}acc)")
            assert _y_var + '_mean' in df.columns
        except:
            _y_var = None

    _agg_var = None
    while _agg_var is None:
        try:
            _agg_var = input(f"Type dependent output aggregation variable (mean,median,min,max)")
            assert _agg_var in ['mean','median','min','max']
        except:
            _agg_var = None
    
    _show_std = None
    while _show_std is None:
        try:
            _show_std = input(f"Show standard deviation (y/n)")
            assert _show_std in ['y','n']
            _show_std = True if _show_std == 'y' else False
        except:
            _show_std = None
    

    
    _x_var = None
    while _x_var is None:
        try:
            _x_var = input(f"Type INdependent variable (e.g. {NOISE_PREFIX}corruption_level)")
            assert _x_var in df.columns
        except:
            _x_var = None

    print("=========================================================================================================")
    print("The following variables have more than one value:")
    _changed_vars = [c for c in df.columns if (not c.startswith(OUTPUT_PREFIX)) and not (c in [_x_var,_y_var])]
    _changed_vars = [c for c in _changed_vars if len(pd.unique(df[c]))>1 ]
    
    _changed_desc = [c + ":   "+  ';'.join(pd.unique(df[c]).astype(np.str)) for c in _changed_vars ]
    print('\n'.join(_changed_desc))
    print("=========================================================================================================")
    print("For each of these variables, you'll have to select one of: ")
    print("l - use as level")
    print("g - use as group (there'll by one plot for each combination of such variables)")
    print("i - ignore (replaces all occurences with first value)")
    
    l_vars = []
    g_vars = []
    for v in _changed_vars:
        ch = 0
        while ch == 0:
            try:
                ch = input(f"Enter the variable type for {v}\n")
                assert ch in ['l','g','i']
                if ch == 'l':
                    l_vars.append(v)
                elif ch == 'g':
                    g_vars.append(v)
                else:
                    df[v] = pd.unique(df[v])[0]
            except:
                ch=0
    fixed_vars = list(set([c for c in df.columns if (not c.startswith(OUTPUT_PREFIX)) and not (c in [_x_var,_y_var])]).difference(g_vars+l_vars))
    fixed_vars_desc =  '\n'.join([c + ":   "+  ';'.join(pd.unique(df[c]).astype(np.str)) for c in fixed_vars ])
    line_plot(df, _x_var,_y_var,_agg_var,g_vars, l_vars, _show_std, OUT_FOLDER,fixed_var_desc=fixed_vars_desc)

def interactive_tex_table(df,OUT_FOLDER):
    
    print("Dataframe columns:")
    print('\n'.join(list(df.columns)))
    
    _out_var = None
    while _out_var is None:
        try:
            _out_var = input(f"Type comma-separated output variables w/ aggregation (e.g. {OUTPUT_PREFIX}acc_mean)")
            _out_var = _out_var.split(',')
            for y in _out_var:
                assert y in df.columns
                assert y.startswith(OUTPUT_PREFIX)
        except:
            _out_var = None

    
    _show_std = None
    while _show_std is None:
        try:
            _show_std = input(f"Use standard deviation  for *_mean output variables (y/n)")
            assert _show_std in ['y','n']
            _show_std = True if _show_std == 'y' else False
        except:
            _show_std = None
    

    
    _x_var = None
    while _x_var is None:
        try:
            _x_var = input(f"Type comma-separated input variables corresponding to VERTICAL divisions variable (e.g. {INPUT_PREFIX}labeled_percent)")
            _x_var = _x_var.split(',')
            for x in _x_var:
                assert x in df.columns
                assert not (x.startswith(OUTPUT_PREFIX))
        except:
            _x_var = None

    print("=========================================================================================================")
    print("The following variables have more than one value:")
    _changed_vars = [c for c in df.columns if (not c.startswith(OUTPUT_PREFIX)) and not (c in _x_var+_out_var)]
    _changed_vars = [c for c in _changed_vars if len(pd.unique(df[c]))>1 ]
    
    _changed_desc = [c + ":   "+  ';'.join(pd.unique(df[c]).astype(np.str)) for c in _changed_vars ]
    print('\n'.join(_changed_desc))
    print("=========================================================================================================")
    print("For each of these variables, you'll have to select one of: ")
    print("h - variable divides the table with horizontal rules")
    print("c - variable does not divide the table with horizontal rule")
    
    print("g - use as group (there'll by one plot for each combination of such variables)")
    print("i - ignore (replaces all occurences with first value)")
    
    c_vars = []
    h_vars = []
    g_vars = []
    for v in _changed_vars:
        ch = 0
        while ch == 0:
            try:
                ch = input(f"Enter the variable type for {v}\n")
                assert ch in ['c','h','g','i']
                if ch == 'c':
                    c_vars.append(v)
                elif ch == 'g':
                    g_vars.append(v)
                elif ch == 'h':
                    h_vars.append(v)
                else:
                    df[v] = pd.unique(df[v])[0]
            except:
                ch=0
    fixed_vars = list(set([c for c in df.columns if (not c.startswith(OUTPUT_PREFIX)) and not (c in _x_var+_out_var)]).difference(g_vars+c_vars+h_vars))
    
    tex_table(df=df,
              y_vars=_out_var,
              t_vars=g_vars, 
              vdiv_vars=_x_var,
              hdiv_vars=h_vars,
              c_vars=c_vars,
             use_std=_show_std,
             OUT_FOLDER=OUT_FOLDER,
            fixed_vars =fixed_vars,
            bold_max=True)
    

  
def create_visualization(FOLDER=CSV_FOLDER,OUT_FOLDER=VIS_FOLDER):
    j_str = '_joined.csv'
    _len = len(j_str)
    fs = [x[:-_len] for x in os.listdir(FOLDER) if x.endswith(j_str) ]
    fs = np.sort(fs)

    """
        -------------------------------------------
            CHOOSE CSVs
        -----------------------------------
    """
    ch = []
    while len(ch) == 0:
        try:
            for i,f in enumerate(fs):
                print(f"{i} - {f}")
            ch = input("Enter comma-separated numbers corresponding to desired experiment\n")
            ch = str.split(ch,',')
            for x in ch:
                assert str.isnumeric(x)
            ch = [osp.join(FOLDER,fs[int(x)]+j_str) for x in ch]
        except:
            ch = []
    print("Selected files:\n {}".format('\n'.join(ch)))
    df = pd.concat([pd.read_csv(x) for x in ch],axis=0,ignore_index=True)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'],axis=1)
    df = df.drop([c for c in df.columns if c.endswith("_values")],axis=1)
    
    """
    ----------------------------------------
        CHOOSE Output folder
    -------------------------------------
    
    """
        
    _subf = None
    while _subf is None:
        try:
            _subf = input(f"Type subfolder name to place visualizations within f{OUT_FOLDER}:\n")
            OUT_FOLDER = osp.join(OUT_FOLDER,_subf)
        except:
            _subf= None
    
    print(f"Will use save visualizations to {OUT_FOLDER}.")
    
    """
        -------------------------------------------
            CHOOSE Visualization
        -----------------------------------
    """
    ch = 0
    while ch == 0:
        try:
            ch = (input("Which visualization?  1-Line plot; 2-LaTeX Table (Use comma-separated values to use multiple)"))
            ch = [int(x) for x in ch.split(',')]
            for x in ch:
                assert x in [1,2]
        except:
            ch = 0
    print(ch)
    if 1 in ch:
        print("Line Plot Creation:\n")
        interactive_line_plot(df,OUT_FOLDER)
    
    if 2 in ch:
        print("LaTeX Table Creation:\n")
        interactive_tex_table(df, OUT_FOLDER)
    
    

if __name__ == "__main__":
    from io import StringIO
    import sys
    
    s = StringIO('\n'.join(['1,2','digit','2','out_acc_labeled_mean,out_acc_unlabeled_mean','y','input_dataset','c','h','i']))
    #sys.stdin = s 
    create_visualization()
    sys.stdin = sys.__stdin__ 
    