import numpy as np

def add_features(dataset):
    """
    Añade nuevas características al dataset:
    - houseAge: Edad de la casa.
    - houseRemodAge: Edad desde la última remodelación.
    - totalBaths: Número total de baños (completos y medios).
    - porchDeckArea: Área total de porches y decks.
    - totalCoveredArea: Área total cubierta.

    Args:
        dataset (pd.DataFrame): Dataset original.

    Returns:
        pd.DataFrame: Dataset con las nuevas características añadidas.
    """
    dataset['houseAge'] = dataset['YrSold'] - dataset['YearBuilt']
    dataset['houseRemodAge'] = dataset['YrSold'] - dataset['YearRemodAdd']
    dataset['totalBaths'] = dataset['BsmtFullBath'] + dataset['FullBath'] + 0.5 * (
        dataset['HalfBath'] + dataset['BsmtHalfBath']
    )
    dataset['porchDeckArea'] = (
        dataset['WoodDeckSF'] + dataset['OpenPorchSF'] + dataset['EnclosedPorch'] +
        dataset['3SsnPorch'] + dataset['ScreenPorch']
    )
    dataset['totalCoveredArea'] = dataset['GrLivArea'] + dataset['TotalBsmtSF']

    return dataset

def drop_features(dataset):
    #drop columns with little que-dar?
    dataset = dataset.drop(columns=['Id','Alley','MasVnrType','BsmtCond','PoolQC','PoolArea','Fence',
                                    'MiscFeature','GarageQual','GarageCond', 'BsmtFinType2'])
    
    #drop columns used in add_features
    dataset = dataset.drop(columns=['YrSold','YearBuilt','YearRemodAdd','BsmtFullBath',
                                   'FullBath','HalfBath','BsmtHalfBath','WoodDeckSF',
                                   'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
                                    'BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','GrLivArea',
                                   'TotalBsmtSF','GarageYrBlt','GarageArea'])
    #drop columns no-representatives

    return dataset

def fill_null_values(dataset):
    string_columns_with_nulls = ['FireplaceQu','GarageFinish','GarageType',
                                 'BsmtExposure','BsmtFinType1','BsmtQual','Electrical']
    dataset[string_columns_with_nulls] = dataset[string_columns_with_nulls].fillna("No")
    
    numeric_columns_with_nulls = ['LotFrontage','MasVnrArea']
    dataset[numeric_columns_with_nulls] = dataset[numeric_columns_with_nulls].fillna(0)

    return dataset

def feature_engineering(dataset):
    dataset_final = dataset.copy()
    dataset_final = add_features(dataset_final)
    dataset_final = drop_features(dataset_final)
    dataset_final = fill_null_values(dataset_final)
    
    if "SalePrice" in dataset_final.columns:
        dataset_final["SalePrice"] = np.log1p(dataset_final["SalePrice"])
        dataset_final=dataset_final.drop([934, 1298, 249, 313, 335, 451, 706, 378, 691, 1182, 297, 1169, 185, 738, 921, 346, 1230, 496, 523])
    
    return dataset_final