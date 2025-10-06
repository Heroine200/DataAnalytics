import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform Exploratory Data Analysis with visualizations
    """
    figs = []
    
    # Age distribution
    plt.figure(figsize=(5, 3))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    figs.append(plt)
    
    # Education Distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x='education_level', data=df)
    plt.title('Education Distribution')
    figs.append(plt)
    
    # Previous employment distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x="previous_employment", data=df)
    plt.title('Previous Employment Distribution')
    figs.append(plt)
    
    # Target variable distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x="suitable_for_employment", data=df)
    plt.title('Employment Suitability Distribution')
    figs.append(plt)
    
    return figs

def print_dataset_info(df):
    """
    Print basic dataset information
    """
    print("Dataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    (df.head())
    
    print("\nDataset Info:")
    df.info()
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    (df.describe())
    
    print("\nTarget variable distribution:")
    print(df['suitable_for_employment'].value_counts())