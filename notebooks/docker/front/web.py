from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List
import pickle
import numpy as np


with open('/home/roman/projects/DS-landscape/data/features/matrix.pkl', 'rb') as f:
    matrix = pickle.load(f)

with open('/home/roman/projects/DS-landscape/data/features/prof_index_to_prof_name.pkl', 'rb') as f:
    prof_index_to_prof_name = pickle.load(f)

with open('/home/roman/projects/DS-landscape/data/features/skill_index_to_corrected.pkl', 'rb') as f:
    skill_index_to_corrected = pickle.load(f)

skill_df = pd.read_csv(
    '/home/roman/projects/DS-landscape/data/features/skills.csv')
prof_df = pd.read_csv(
    '/home/roman/projects/DS-landscape/data/features/prof.csv')

m_max = np.max(matrix, axis=0)
m_min = np.min(matrix, axis=0)
normalized_matrix = matrix * 100.0 / matrix.sum(axis=0)

df_plot_polar = pd.DataFrame(normalized_matrix).rename(
    columns=prof_index_to_prof_name)
df_plot_polar['skill_id'] = skill_index_to_corrected.keys()
df_plot_polar['Навык'] = skill_index_to_corrected.values()
assert np.all([skill_index_to_corrected[df_plot_polar['skill_id'][i]] ==
              df_plot_polar['Навык'][i] for i in skill_index_to_corrected.keys()])


def plot_polar_graph(df: pd.DataFrame, prof_list: List[str], intersect_skills: bool = False, top_n: int = 30) -> go.Figure:
    skill_set = []

    for prof_name in prof_list:
        df_copy = df[[prof_name, 'Навык']].copy()
        prof_skill = df_copy.sort_values(by=prof_name, ascending=False)[
            'Навык'][:top_n].values

        if intersect_skills:
            if len(skill_set) == 0:
                skill_set = prof_skill
            skill_set = [x for x in prof_skill if x in skill_set]
        else:
            skill_set = skill_set + \
                [x for x in prof_skill if x not in skill_set]

    df_copy = df[df['Навык'].isin(skill_set)]

    df_copy = df_copy.sort_values(by=prof_list[0], ascending=False)

    fig = go.Figure()

    for _, prof_name in enumerate(prof_list):

        fig.add_trace(go.Scatterpolar(
            r=df_copy[prof_name],
            theta=df_copy['Навык'],
            fill='toself',
            name=prof_name
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            ),
            showlegend=True
        )

    return fig


def plot_skill_scatter(df: pd.DataFrame, width: int = 800, height: int = 600) -> go.Figure:
    fig = px.scatter(df, x='x', y='y',
                     color='Профессия', hover_name='Навык',
                     hover_data={'x': False, 'y': False,
                                 'size': False, 'Профессия': False},
                     size='size', category_orders={'Профессия': profession_list},
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     title=None, width=width, height=height)

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


profession_list = [
    'ML инженер', 'Инженер данных', 'Системный аналитик', 'Продуктовый аналитик',
    'Computer Vision', 'Data Scientist', 'NLP', 'Администратор баз данных',
    'Аналитик BI', 'Аналитик', 'Бизнес-аналитик', 'Big Data', 'Аналитик данных'
]

polar_fig = plot_polar_graph(
    df_plot_polar, prof_list=profession_list, intersect_skills=True, top_n=25)

df_plot_scatter = pd.read_csv('data/plot_df/best.csv')
scatter_fig = plot_skill_scatter(df_plot_scatter, width=800, height=600)

app = Dash(__name__)


preset_bar_style = {
    'display': 'flex',
    'justify-content': 'flex-start',
    'margin-bottom': '20px'
}

preset_dropdown_style = {
    'width': '300px',
    'margin-right': '10px'
}


app.layout = html.Div(
    children=[
        html.H4('Интерактивная карта навыков (можно выбрать разные режимы в выпадающих списках и кликать категории в легенде)'),
        html.Div(
            id='preset-bar',
            style=preset_bar_style,
            children=[
                html.Div(
                    style={'flex': '1', 'margin-right': '10px'},
                    children=[
                        dcc.Dropdown(
                            id='preset-dropdown',
                            options=[
                                {'label': 'DS и ML', 'value': ['ML инженер', 'Data Scientist']},
                                {'label': 'NLP и CV', 'value': ['NLP', 'Computer Vision']},
                                {'label': 'Аналитик', 'value': ['Системный аналитик', 'Продуктовый аналитик', 'Аналитик BI',
                                                               'Аналитик', 'Бизнес-аналитик', 'Аналитик данных']},
                                {'label': 'Данные', 'value': ['Администратор баз данных', 'Big Data', 'Инженер данных']}
                            ],
                            placeholder='Выберите пресет',
                            value=['ML инженер', 'Data Scientist'],
                        ),
                    ]
                ),
                html.Div(
                    style={'flex': '1', 'margin-right': '10px'},
                    children=[
                        dcc.Checklist(
                            options=[
                                {'label': 'Круговая диаграмма', 'value': 'polar'},
                                {'label': 'Диаграмма рассеивания', 'value': 'scatter'}
                            ],
                            value=['polar', 'scatter'],
                            id='chart-checkboxes',
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                        ),
                    ]
                ),
                html.Div(
                    style={'flex': '1'},
                    children=[
                        dcc.Slider(
                            id='num-skills-slider',
                            min=25,
                            max=40,
                            step=5,
                            value=30,
                            marks={i: str(i) for i in range(25, 41, 5)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ]
                ),
            ]
        ),
        html.Div(id='chart-container', style={'display': 'flex'})
    ]
)



@app.callback(
    Output('chart-container', 'children'),
    Input('preset-dropdown', 'value'),
    Input('chart-checkboxes', 'value'),
    Input('num-skills-slider', 'value')
)
def update_chart(selected_preset, selected_charts, num_skills):
    charts = []
    if selected_preset:
        prof_list = selected_preset
    else:
        prof_list = ['ML инженер', 'Data Scientist']
    
    if 'polar' in selected_charts:
        polar_fig = plot_polar_graph(
            df_plot_polar, prof_list=prof_list, intersect_skills=True, top_n=num_skills)
        charts.append(dcc.Graph(id='polar-plot', figure=polar_fig))
    if 'scatter' in selected_charts:
        scatter_fig = plot_skill_scatter(
            df_plot_scatter, width=800, height=600)
        charts.append(dcc.Graph(id='scatter-plot', figure=scatter_fig))
    return charts


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
