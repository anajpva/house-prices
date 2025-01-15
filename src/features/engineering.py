import numpy as np

# Add new features to the dataset
def add_features(dataset):
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
    
# Drop unnecessary features from the dataset
def drop_features(dataset):
    # Drop columns with little information or high correlation with others
    dataset = dataset.drop(columns=['Id','Alley','MasVnrType','BsmtCond','PoolQC','PoolArea','Fence',
                                    'MiscFeature','GarageQual','GarageCond', 'BsmtFinType2'])
    
    # Drop columns used to create new features
    dataset = dataset.drop(columns=['YrSold','YearBuilt','YearRemodAdd','BsmtFullBath',
                                   'FullBath','HalfBath','BsmtHalfBath','WoodDeckSF',
                                   'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
                                    'BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','GrLivArea',
                                   'TotalBsmtSF','GarageYrBlt','GarageArea'])

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
        # Normalize the selling price distribution
        dataset_final["SalePrice"] = np.log1p(dataset_final["SalePrice"])
        
        # Remove rows based on specific limit criteria to clean the dataset
        dataset_final=dataset_final.drop([934, 1298, 249, 313, 335, 451, 706, 378, 691, 1182, 297, 1169, 185, 738, 921, 346, 1230, 496, 523])
    
    return dataset_final