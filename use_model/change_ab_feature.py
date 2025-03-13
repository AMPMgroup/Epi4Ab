import warnings

def extract_ab_input_feature(ab_feature:dict, ex_continuous_columns, ex_onehot_vh_columns, ex_onehot_vl_columns):
    # Check continuous columns
    ab_continuous_columns = list(ab_feature['continuous'].keys())
    # excluded_columns = [x for x in ex_continuous_columns if x not in ab_continuous_columns]
    extra_columns = [x for x in ab_continuous_columns if x not in ex_continuous_columns]
    # if excluded_columns:
    #     raise ValueError(f'{excluded_columns} is required.')
    if extra_columns:
        warnings.warn(f'{extra_columns} will not be included in this run.')
    # Check one-hot columns
    ab_vh_type = ab_feature['vh_type']
    ab_vl_type = ab_feature['vl_type']
    if ab_vh_type not in ex_onehot_vh_columns:
        raise ValueError(f'{ab_vh_type} is not in {ex_onehot_vh_columns}')
    if ab_vl_type not in ex_onehot_vl_columns:
        raise ValueError(f'{ab_vl_type} is not in {ex_onehot_vl_columns}')
    
    # ab_continuous_values = [ab_feature['continuous'][x] for x in ex_continuous_columns]
    ab_continuous_values = [ab_feature['continuous'][x] for x in ab_continuous_columns]
    ab_vh_values = [1 if x == ab_vh_type else 0 for x in ex_onehot_vh_columns]
    ab_vl_values = [1 if x == ab_vl_type else 0 for x in ex_onehot_vl_columns]
    return ab_continuous_columns, ab_continuous_values, ab_vh_values, ab_vl_values