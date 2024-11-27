from datetime import datetime
from plotly.graph_objects import FigureWidget
from IPython.display import display
from ipywidgets import HBox, VBox, Box, fixed, interactive_output, Output

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
import csv
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objects as go
import numpy as np

box_layout = widgets.Layout(
    display='flex',
    flex_flow='row wrap',  # Définit l'orientation et l'autorise à passer à la ligne
    justify_content='space-around',  # Espacement autour des items
    align_items='center',  # Aligne les items au centre verticalement
    width='100%',  # Largeur du conteneur
)

def generate_widget_scatter_matrix(df, columns=None, width=None, height=None, title=''):
    if columns is None:
        columns = df.columns
    
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        title=title,
    )
    
    fig.update_layout(autosize=True, width=width, height=height)

    return FigureWidget(fig)

def generate_widget_corr(df, columns=None, columns_to_empty=[], width=None, height=None, title=''):
    if columns is None:
        columns = df.columns
        
    df_corr = df.loc[:, columns].corr()

    for c in columns:
        df_corr.loc[c, c] = 0
        
    for c1 in columns_to_empty:
        for c2 in columns_to_remove:
            df_corr.loc[c1, c2] = 0
    
    fig = px.imshow(
        df_corr,
        color_continuous_scale="rdbu",
        title=str(title),
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )

    fig.update_layout(autosize=True, width=width, height=height)
    fig.update_traces(textfont_size=16)

    return FigureWidget(fig)

def generate_widget_corr_column(df, column_value, columns_value=None, columns_all=None, columns_to_empty=[], values=None, width=None, height=None):
    widgets = list()
    
    columns_filtered = columns_all if columns_value is None \
        else list(filter(lambda c: c not in columns_value, columns_all)) 

    if values is None:
        values = df[column_value].unique()

    if columns_all is None:
        columns_all = df.columns
       
    for value in values:
        df_filtered = df.loc[df[column_value] == value, columns_all]
        
        widgets.append(
            generate_widget_corr_class(
                df_filtered, 
                title=value, 
                columns=columns_filtered,
                columns_to_empty=columns_to_empty,
                width=width,
                height=height,
            )
        )

    return widgets

def generate_widget_scatter(df, x, y, point=None, width=None, height=None, title='', trendline='lowess', frac=0.9):
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        title=str(title),
        trendline=trendline,
        trendline_options=dict(frac=frac) if trendline else dict(),
    )
    
    fig.update_traces(marker=dict(size=point))
    fig.update_layout(autosize=True, width=width,height=height)

    return FigureWidget(fig)

def generate_widget_scatter_column(df, column_value, x, y, values=None, point=None, width=None, height=None, trendline='lowess', frac=0.9):
    widgets = list()

    if values is None:
        values = df[column_value].unique()
       
    for value in values:
        df_filtered = df.loc[df[column_value] == value]
        
        widgets.append(
            generate_widget_scatter(
                df_filtered, 
                title=value, 
                x=x,
                y=y,
                point=point,
                width=width,
                height=height,
                trendline=trendline,
                frac=frac,
            )
        )

    return widgets

def generate_widget_histo(df, x, y, width=None, height=None, title=''):
    fig = px.histogram(
        df, 
        x=x, 
        y=y,
        title=str(title),
    )
    
    fig.update_layout(autosize=True, width=width, height=height)

    return FigureWidget(fig)

def generate_widget_histo_column(df, column_value, x, y, values=None, width=500, height=500):
    widgets = list()

    if values is None:
        values = df[column_value].unique()
       
    for value in values:
        df_filtered = df.loc[df[column_value] == value]
        
        widgets.append(
            generate_widget_histo(
                df_filtered, 
                title=value,
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    return widgets

def generate_widget_box(df, x, y, width=None, height=None, title=''):
    fig = px.box(
        df, 
        x=x, 
        y=y,
        title=str(title),
    )
    
    fig.update_layout(autosize=True, width=width, height=height)

    return FigureWidget(fig)

def generate_widget_box_column(df, column_value, x, y, values=None, width=None, height=None):
    widgets = list()

    if values is None:
        values = df[column_value].unique()
       
    for value in values:
        df_filtered = df.loc[df[column_value] == value]
        
        widgets.append(
            generate_widget_box(
                df_filtered, 
                title=value,
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    return widgets

def generate_widget_table(df):
    out = Output()

    with out:
        display(df)

    return out

def generate_box(children, layout=box_layout):
    return Box(
        children=children, 
        layout=layout
    )

def regression_lineaire(df, X_columns, y_column):
    X_name = '/'.join(X_columns)
    
    df_cleaned = df[~(df[y_column].isnull())]
    for X_column in X_columns:
        df_cleaned = df_cleaned[~(df_cleaned[X_column].isnull())]

    df_cleaned = df_cleaned.reset_index()

    # Variables explicatives (X) et cible (y)
    X = df_cleaned[X_columns]  # Les colonnes explicatives
    y = df_cleaned[y_column]  # La colonne cible
    
    # Diviser les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model_1 = LinearRegression().fit(X_train, y_train)
    model_intercept = round(model_1.intercept_, 2)
    model_coef = round(model_1.coef_[0], 2)
    
    print(f"On s'attend à avoir environ {model_intercept}° pour un {X_name} de 0.")
    print(f"Chaque {X_name} supplémentaire ajoute environ {model_coef} {y_column}, ou on ajoute {int(model_coef*100)}{y_column} tous les 100 {X_name}.")

    r2_train = model_1.score(X_train, y_train)

    train_pred_model_1 = model_1.predict(X_train)
    
    mse_train = mean_squared_error(y_true=y_train, y_pred=train_pred_model_1)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=train_pred_model_1)
    
    print(f"""
    Le coefficient de détermination, R² ~= {round(r2_train,2)}, signifie que le modèle explique {int(r2_train*100)}% de la variance.
    L'erreur moyenne quadratique MSE vaut environ {round(mse_train,2)}. Plus la MSE est faible, meilleur est le modèle.
    L'erreur moyenne absolue MAE se situe vers {round(mae_train, 2)}, ce qui est interprétable : cela signifie qu'en moyenne, l'erreur de prédiction sera d'environ {round(mae_train, 2)} {y_column}.
    """)

    regression_lineaire_curve(model_1, X_train, y_train)

    return regression_lineaire_test(model_1, df_cleaned, X_columns, y_column, r2_train, mse_train, mae_train)

def regression_lineaire_curve(model_1, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model_1, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    
    # Calculer les erreurs moyennes pour chaque taille d'échantillon d'entraînement
    # (négatif car l'erreur est retournée négative)
    train_errors = -train_scores.mean(axis=1)
    test_errors = -test_scores.mean(axis=1)
    
    learning_curve_df = pd.DataFrame({
        'train_size': train_sizes,
        'train_error': train_errors,
        'test_error': test_errors
    })
    
    fig = px.line(
        learning_curve_df, 
        x='train_size', 
        y=['train_error', 'test_error'],
        labels={'train_size': 'Taille de l\'échantillon d\'entrainement', 'value': 'Erreur MSE'},
        title='Learning Curve pour une régression linéaire'
    )
    
    fig.show()

def regression_lineaire_test(model_1, df, X_columns, y_column, r2_train, mse_train, mae_train):  
    df_cleaned = df[~(df[y_column].isnull())]
    for X_column in X_columns:
        df_cleaned = df_cleaned[~(df_cleaned[X_column].isnull())]

    df_cleaned = df_cleaned.reset_index()
        
    X_test = df_cleaned[X_columns]  # Les colonnes explicatives
    y_test = df_cleaned[y_column]  # La colonne cible
    
    test_pred_model_1 = model_1.predict(X_test)

    sample_count_min = min(X_test.shape[0], y_test.shape[0], len(test_pred_model_1))
    sample_count = min(200, len(test_pred_model_1))
    
    X_test_sample = X_test.sample(n=sample_count, random_state=sample_count_min)
    y_test_sample = y_test.loc[X_test_sample.index]
    pred_sample = test_pred_model_1[X_test_sample.index - 1] # Les index commencent à 1
    
    for X_column in X_columns:
        df_column = pd.DataFrame({
          X_column:           X_test_sample[X_column],
          y_column + '_true': y_test_sample,
          y_column + '_pred': pred_sample,
        })

        fig = px.scatter(
            df_column, 
              x=X_column, 
              y=[y_column + '_true', y_column + '_pred'], 
              title=f"Prédiction {y_column} / {X_column}",
            labels={'value': y_column},
            trendline='lowess'
        )
        
        fig.update_layout(autosize=True, height=400)
        fig.show()

    r2_test = model_1.score(X_test, y_test)

    mse_test = mean_squared_error(y_true=y_test, y_pred=test_pred_model_1)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=test_pred_model_1)
    
    print(f"""
    Coefficient de détermination, R² : (à augmenter)
    - Train = {r2_train}
    - Test =  {r2_test}
    
    L'erreur moyenne quadratique MSE : (à réduire)
    - Train = {mse_train}
    - Test =  {mse_test}
    
    L'erreur moyenne absolue MAE : (à réduire)
    - Train = {mae_train}
    - Test =  {mae_test}
    
    Un sur-apprentissage se détecte si les métriques sont trop différentes
    """)
    

    return model_1, r2_train, mse_train, mae_train