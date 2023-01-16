from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import requests
import json


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
header = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
    }
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([], className='vl', style={'padding': 10, 'flex': 1}),
    html.Div([
        html.H1("Predict logical relationship", className='h_font'),
        html.Label("Input text 1"),
        dcc.Textarea(
            id='textarea-1-state', 
            value=''
            ),
        html.Label("Input text 2"),
        dcc.Textarea(
            id='textarea-2-state', 
            value=''
            ),
        html.Br(),
        html.Button(
            id='submit-button-state', 
            n_clicks=0, 
            children='Submit',
            style={'background-color': "blue", 'color': 'white'}
            ),
        html.Div(id="info-message", style={'color': "red"}),
        dcc.Graph(id='prob-bar-graph', figure = blank_fig())
    ], style={'padding': 10, 'flex': 1}),

    html.Div([
        html.H1("Custom zero-short text classification", className='h_font'),
        html.Label("Input your own classes one per line"),
        dcc.Textarea(
            id='textarea-classes-state', 
            value=''
            ),
        html.Label("Your text for classification"),
        dcc.Textarea(
            id='textarea-text-state', 
            value=''
            ),
        html.Br(),
        html.Button(
            id='submit-button-2-state', 
            n_clicks=0, 
            children='Submit',
            style={'background-color': "blue", 'color': 'white'}
            ),
        html.Div(id="info-message-2", style={'color': "red"}),
        dcc.Graph(id='prob-bar-graph-2', figure = blank_fig())
    ], style={'padding': 10, 'flex': 1})
], style={'display': 'flex', 'flex-direction': 'row'})

@app.callback(
    Output('info-message', 'children'),
    Output('prob-bar-graph', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('textarea-1-state', 'value'),
    State('textarea-2-state', 'value')
    )
def update_logical_pred(n_clicks, input1, input2):
    out_str = ''
    input = {'text1': input1, 'text2': input2}
    response = requests.post(
        "http://127.0.0.1:9090/api/v1.0/pred_logical_rel",
        data = json.dumps(input),
        headers= header
        )

    if response.status_code == 400:
        if n_clicks != 0:
            out_str = 'Please, input text in the missing cells'
        return out_str, blank_fig()

    out = response.json()      
    df_out = pd.DataFrame({'type': out.keys(), 'probability': out.values()})
    fig = px.bar(
        df_out, x='type', y='probability', text_auto='.2f',
        title='Probabiity distribution', template = 'seaborn'
    )
    fig.update_layout(font={'size': 16}, transition_duration=500)
    return out_str, fig

@app.callback(
    Output('info-message-2', 'children'),
    Output('prob-bar-graph-2', 'figure'),
    Input('submit-button-2-state', 'n_clicks'),
    State('textarea-classes-state', 'value'),
    State('textarea-text-state', 'value'),
    )
def update_zero_shot_pred(n_clicks, input_classes, input_text):
    out_str = ''
    input = {'classes': input_classes, 'text': input_text}
    response = requests.post(
        "http://127.0.0.1:9090/api/v1.0/pred_zero_shot",
        data = json.dumps(input),
        headers= header
        )

    if response.status_code == 400:
        if n_clicks != 0:
            out_str = 'It looks like you forgot to input text or your own classes'
        return out_str, blank_fig()

    out = response.json()   
    df_out = pd.DataFrame({'classes': out.keys(), 'probability': out.values()})
    fig = px.bar(
        df_out, x='classes', y='probability', text_auto='.2f',
        title='Probabiity distribution', template = 'seaborn'
    )
    fig.update_layout(font={'size': 16}, transition_duration=500)
    return '', fig

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='9999', debug=True)
