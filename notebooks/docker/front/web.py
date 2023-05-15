from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly
import pandas as pd

def plot_skill_map(df: pd.DataFrame, width=1000, height=600) -> plotly.graph_objs.Figure:
    fig = px.scatter(df, x='x', y='y',
                    color='Профессия', hover_name='Навык', 
                    hover_data= {'x':False, 'y':False, 'size':False, 'Профессия': False},
                    size='size',
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title = None, width=width, height=height)

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')), 
                  selector=dict(mode='markers'))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

def plot_prof_map(df: pd.DataFrame, width=1000, height=600) -> plotly.graph_objs.Figure:
    fig = px.scatter(df, x='x', y='y',
        color='Профессия', hover_name='Навык', 
        hover_data= {'x':False, 'y':False, 'size':False, 'Профессия': False},
        title = None, width=width, height=height)

    fig.update_traces(marker=dict(size=20, line=dict(width=0.5, color='DarkSlateGrey')), 
                      selector=dict(mode='markers'))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Интерактивная карта навыков (можно выбрать разные режимы в выпающих списках и кликать категории в легенде)'),
    dcc.Dropdown(['50 новыков каждой профессии', '100 новыков каждой профессии', '200 новыков каждой профессии'], 
                 '100 новыков каждой профессии', id='top-dropdown'),
    dcc.Dropdown(['1-вый алгоритм', '2-вый алгоритм', '3-ий алгоритм'], 
                 '1-вый алгоритм', id='alg-dropdown'),
    dcc.Graph(id="scatter-plot"),

])



@app.callback(
    Output("scatter-plot", "figure"), 
    Input("top-dropdown", "value"),
    Input("alg-dropdown", "value"))
def update_bar_chart(top: str, alg: str):
    n = 'none'
    a = 'tnse'

    if top.startswith('50 '):
        top = 50
    elif top.startswith('100 '):
        top = 100
    elif top.startswith('200 '):
        top = 200
    else:
        raise ValueError('top parametr invalid')

    if alg.startswith('1-'):
        n = 'none'
        a = 'tnse'
    elif alg.startswith('2-'):
        n = 'prof'
        a = 'tnse'
    elif alg.startswith('3-'):
        n = 'skill'
        a = 'als'
    else:
        raise ValueError('alg parametr invalid')

    df_plot_skill = pd.read_csv(f'data/plot_df/{a}-{n}-{top}-skill.csv')
    fig = plot_skill_map(df_plot_skill, width=1200, height=750)
    return fig 

if __name__ == '__main__':
    # from werkzeug.serving import run_simple
    print('start from main front')
    # run_simple('127.0.0.1', 8050, server)
    app.run(host='0.0.0.0')