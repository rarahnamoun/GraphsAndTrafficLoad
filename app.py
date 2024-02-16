import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import networkx as nx

app = dash.Dash()

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

app.layout = html.Div([
    html.H1('Graph Visualization'),
    html.Label('Graph Type'),
    dcc.Dropdown(
        id='graph-type-dropdown',
        options=[
            {'label': 'Barabasi-Albert', 'value': 'barabasi_albert'},
            {'label': 'Erdos-Renyi', 'value': 'erdos_renyi'}
        ],
        value='barabasi_albert'
    ),
    html.Label('Number of Nodes'),
    dcc.Input(id='nodes-input', type='number', value=20),
    dcc.Graph(id='graph-output')

])





def generate_graph(graph_type, num_nodes):
    G = nx.Graph()

    if graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_nodes, 5)  # Change 5 to the desired number of edges to attach from a new node
    elif graph_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(num_nodes, 0.2)  # Change 0.2 to the desired probability of edge creation

    pos = nx.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = 'Name: ' + str(adjacencies[0]) + '<br># of connections: ' + str(len(adjacencies[1]))
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph of ' + str(num_nodes) + ' rules',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig


@app.callback(
    Output('graph-output', 'figure'),
    [Input('graph-type-dropdown', 'value'),
     Input('nodes-input', 'value')])
def update_graph(graph_type, nodes):
    graph = generate_graph(graph_type, int(nodes))
    return graph


if __name__ == '__main__':
    app.run_server(debug=True)
