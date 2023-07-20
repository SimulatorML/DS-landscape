from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List
import pickle
import numpy as np


with open('data/polar/matrix.pkl', 'rb') as f:
    matrix = pickle.load(f)

with open('data/polar/prof_index_to_prof_name.pkl', 'rb') as f:
    prof_index_to_prof_name = pickle.load(f)

with open('data/polar/skill_index_to_corrected.pkl', 'rb') as f:
    skill_index_to_corrected = pickle.load(f)

skill_df = pd.read_csv(
    'data/polar/skills.csv')
prof_df = pd.read_csv(
    'data/polar/prof.csv')

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
    color_list = [
        '#F8A19F', '#AA0DFE', '#3283FE', '#1CBE4F', '#C4451C', '#F6222E', 
        '#FE00FA', '#325A9B', '#FEAF16', 
        '#90AD1C', '#2ED9FF', '#B10DA1',
         '#909090', '#FBE426',
        '#FA0087', '#C075A6', '#FC1CBF'
    ]

    prof_order = [
        'Data Scientist', 'ML инженер', 'Computer Vision', 'NLP',
        'Инженер данных', 'Big Data', 'Администратор баз данных', 'Аналитик данных',
        'Аналитик', 'Бизнес-аналитик', 'Продуктовый аналитик', 'Аналитик BI',
        'Системный аналитик' ]
    
    fig = px.scatter(df, x='x', y='y',
                     color='Профессия', hover_name='Навык',
                     hover_data={'x': False, 'y': False,
                                 'size': False, 'Профессия': False},
                     size='size', category_orders={'Профессия': prof_order},
                     color_discrete_sequence=color_list,
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

df_plot_scatter = pd.read_csv('data/scatter/best.csv')
scatter_fig = plot_skill_scatter(df_plot_scatter, width=1000, height=600)

app = Dash(__name__)

dropdownvalue_to_preset = {
    0: ['ML инженер', 'Data Scientist'],
    1: ['NLP', 'Computer Vision'],
    2: ['Аналитик', 'Аналитик данных', 'Системный аналитик', 'Продуктовый аналитик', 'Аналитик BI',
                                                               'Бизнес-аналитик'],
    3: ['Инженер данных','Администратор баз данных', 'Big Data']
}


app.layout = html.Div(
    children=[
        html.H1('Добро пожаловать в SkillMap: Ваш путеводитель в мир навыков и профессий'),
        html.Div(id='descr',style={'font-size': '0.9em'}, children=[
            html.P("SkillMap - инновационный сервис, созданный специально для тех, кто хочет успешно развиваться в современном образовательном и профессиональном пространстве. Независимо от того, являетесь ли вы студентом, кандидатом, специалистом, работодателем, менеджером по персоналу, менеджером продукта или автором учебных курсов, SkillMap поможет вам осуществить вашу учебную и профессиональную стратегию наиболее эффективно"),
            html.P("Мы понимаем ваши потребности и проблемы. Вы стремитесь расширить свои навыки, понять, какие навыки наиболее востребованы и как их освоить. SkillMap предлагает вам уникальный инструмент для ориентирования в мире навыков и профессий."),
            html.P("При посещении нашей веб-страницы вы получите уникальную карту, которая наглядно демонстрирует наиболее востребованные навыки и их взаимосвязи. Вы сможете увидеть, какие навыки и специальности наиболее востребованы, и как они взаимосвязаны между собой. Наша визуализация интерактивная и позволяет вам исследовать различные пути и варианты развития. Мы уверены, что наш сервис поможет вам увидеть широкий спектр возможностей и принять взвешенные решения в отношении ваших навыков и карьеры."),
            html.P("SkillMap - ваш надежный партнер в путешествии по миру навыков Data Science. Подключайтесь к нашей платформе сегодня и начинайте достигать новых высот в своей учебе и карьере!"),

            html.P("Выберите пресет профессий и количество навыков для круговой диаграммы. Или отметьте интересующие профессиии на точечной диаграмме. Двойной клик на профессии отключит все остальные."),

            html.Br()
        ]),

        html.Div(
            style={'display': 'flex'},
            children=[
                html.Div(
                    id='polar',
                    style={
                        'width': '33%',
                        'justify-content': 'flex-start',
                        'margin-bottom': '20px'
                    },
                    children=[
                        html.Div(
                            style={'flex': '1', 'margin-right': '10px'},
                            children=[
                                dcc.Dropdown(
                                    id='preset-dropdown',
                                    options=[
                                        {'label': 'DS и ML', 'value': 0},
                                        {'label': 'NLP и CV', 'value': 1},
                                        {'label': 'Аналитик', 'value': 2},
                                        {'label': 'Данные', 'value': 3}
                                    ],
                                    value=0,
                                    clearable=False
                                ),
                            ]
                        ),
                        html.Div(
                            style={'flex': '1'},
                            children=[
                                dcc.Slider(
                                    id='num-skills-slider',
                                    min=4,
                                    max=40,
                                    step=4,
                                    value=12,
                                    marks=None,
                                )
                            ]
                        ),
                        html.Div(id='chart-container', style={'display': 'flex'})
                    ]
                ),
                
                html.Div(
                    style={'flex': '1'},
                    children=[dcc.Graph(id='scatter-plot', figure=scatter_fig)]
                )
            ]
        )

        

    ]
)



@app.callback(
    Output('chart-container', 'children'),
    Input('preset-dropdown', 'value'),
    #Input('chart-checkboxes', 'value'),
    Input('num-skills-slider', 'value')
)
def update_chart(selected_preset, 
                 #selected_charts, 
                 num_skills):
    charts = []
    if selected_preset:
        prof_list = dropdownvalue_to_preset[selected_preset]
    else:
        prof_list = dropdownvalue_to_preset[0]
    
    polar_fig = plot_polar_graph(
        df_plot_polar, prof_list=prof_list, intersect_skills=True, top_n=num_skills)
    charts.append(dcc.Graph(id='polar-plot', figure=polar_fig, style={'width': '90vh'}))

    return charts


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
