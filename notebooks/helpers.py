from datetime import datetime
from plotly.graph_objects import FigureWidget
from IPython.display import display
from ipywidgets import HBox, VBox, Box, fixed, interactive_output

import os
import csv
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objects as go

box_layout = widgets.Layout(
    display='flex',
    flex_flow='row wrap',  # Définit l'orientation et l'autorise à passer à la ligne
    justify_content='space-around',  # Espacement autour des items
    align_items='center',  # Aligne les items au centre verticalement
    width='100%',  # Largeur du conteneur
)

def generate_widget_corr(df, title, columns, columns_to_remove=[], size=700):
    df_corr = df.loc[:, columns].corr()

    for c in columns:
        df_corr.loc[c, c] = 0
        
    for c1 in columns_to_remove:
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

    fig.update_layout(width=size, height=size)
    fig.update_traces(textfont_size=16)
    fig.update_xaxes(mirror=False)

    return FigureWidget(fig)

def generate_widget_scatter(df, title, x, y, point=30, width=500, height=500, withTrendline=False):
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        title=str(title),
        trendline='lowess' if withTrendline else None,
        trendline_options=dict(frac=0.9) if withTrendline else dict(),
    )
    
    fig.update_traces(marker=dict(size=point))
    fig.update_layout(width=width,height=height)

    return FigureWidget(fig)

def generate_widget_scatter_all(df, columns, point=30, width=500, height=500, withTrendline=False):
    widgets = list()
    graphs = list()
    
    for c in columns:
        for r in columns:
            if c != r and f'{c}-{r}' not in graphs and f'{r}-{c}' not in graphs:
                graphs.append(f'{c}-{r}')
                
                widgets.append(
                    generate_widget_scatter(df, f'{c} / {r}', c, r, point, width, height, withTrendline)
                )

    return widgets

def generate_widget_histo(df, title, x, y, width=500, height=500):
    fig = px.histogram(
        df, 
        x=x, 
        y=y,
        title=str(title),
    )
    
    fig.update_layout(width=width, height=height)

    return FigureWidget(fig)

def generate_widget_box(df, title, x, y, width=500, height=500):
    fig = px.box(
        df, 
        x=x, 
        y=y,
        title=str(title),
    )
    
    fig.update_layout(width=width, height=height)

    return FigureWidget(fig)

def generate_widget_corr_zone(df, zone, columns, columns_to_remove, size=700):
    df_filtered = df.loc[df['zone'] == zone, columns]

    return generate_widget_corr(df_filtered, zone, columns, columns_to_remove, size)

def generate_widget_scatter_col(df, col, value, x, y, point=30, width=500, height=700, withTrendline=False):
    df_filtered = df.loc[df[col] == value]

    return generate_widget_scatter(df_filtered, value, x, y, point, width, height, withTrendline)

def generate_widget_histo_col(df, col, value, x, y, width=500, height=700):
    df_filtered = df.loc[df[col] == value]

    return generate_widget_histo(df_filtered, value, x, y, width, height)

def generate_widget_box_col(df, col, value, x, y, width=500, height=700):
    df_filtered = df.loc[df[col] == value]

    return generate_widget_box(df_filtered, value, x, y, width, height)

def generate_widget_corr_class(df, class_name, columns, columns_to_remove, size=700):
    df_filtered = df.loc[df['class'] == class_name, columns]

    return generate_widget_corr(df_filtered, class_name, columns, columns_to_remove, size)

def generate_widget_corr_hour(df, hour, columns, columns_to_remove, size=700):
    df_filtered = df.loc[df['hour'] == hour, columns]

    return generate_widget_corr(df_filtered, hour, columns, columns_to_remove, size)

def generate_widget_corr_by_class(df, columns, columns_to_remove, size=700):
    widgets = list()
       
    for class_name in classes_all:
        widgets.append(
            generate_widget_corr_class(
                df, 
                class_name=class_name, 
                columns=list(filter(lambda c: c != 'class' and c != 'class_index', columns)),
                columns_to_remove=columns_to_remove,
                size=size
            )
        )

    return widgets

def generate_widget_corr_by_zone(df, columns, columns_to_remove, size=500):
    widgets = list()
       
    for zone in zones_all:
        widgets.append(
            generate_widget_corr_zone(
                df, 
                zone, 
                list(filter(lambda c: c != 'zone' and c != 'zone_id', columns)),
                columns_to_remove,
                size=size,
            )
        )

    return widgets

def generate_widget_scatter_by_col(df, col, x, y, point=30, width=500, height=500, withTrendline=False):
    widgets = list()
       
    for value in df[col].unique():
        widgets.append(
            generate_widget_scatter_col(
                df, 
                col,
                value, 
                x,
                y,
                point=point,
                width=width,
                height=height,
                withTrendline=withTrendline
            )
        )

    return widgets

def generate_widget_histo_by_col(df, col, x, y, width=500, height=500):
    widgets = list()
       
    for value in df[col].unique():
        widgets.append(
            generate_widget_histo_col(
                df, 
                col,
                value, 
                x,
                y,
                width=width,
                height=height,
            )
        )

    return widgets

def generate_widget_box_by_col(df, col, x, y, width=500, height=500):
    widgets = list()
       
    for value in df[col].unique():
        widgets.append(
            generate_widget_box_col(
                df, 
                col,
                value, 
                x,
                y,
                width=width,
                height=height,
            )
        )

    return widgets

def generate_widget_corr_by_hour(df, columns, columns_to_remove, size=500):
    widgets = list()
       
    for hour in hours_all:
        widgets.append(
            generate_widget_corr_by_hour(
                df, 
                hour, 
                list(filter(lambda c: c != 'hour', columns)),
                columns_to_remove,
                size=size,
            )
        )

    return widgets

def generate_box(children, layout=box_layout):
    return Box(
        children=children, 
        layout=layout
    )