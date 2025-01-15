from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.compose import  ColumnTransformer


num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ore_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ore', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

def pipeline(num_cols, ore_cols, ohe_cols):

    col_trans = ColumnTransformer(transformers=[
        ('num_pipe', num_pipeline, num_cols),
        ('ore_pipe', ore_pipeline, ore_cols),
        ('ohe_pipe', ohe_pipeline, ohe_cols),
        ],
        remainder='passthrough', 
        n_jobs=-1
        )

    pipeline = Pipeline(steps=[
        ('preprocessing', col_trans)
    ])

    return pipeline
