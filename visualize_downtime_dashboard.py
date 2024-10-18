import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap'
])

resampled_data_with_ano = pd.read_csv('processed_data.csv')


# Extract measurement options (Stage 1 and Stage 2) for the dropdown
measurement_options = [
    {'label': f'Stage 1 Output Measurement {i}', 'value': f'Stage1.Output.Measurement{i}'}
    for i in range(15)
] + [
    {'label': f'Stage 2 Output Measurement {i}', 'value': f'Stage2.Output.Measurement{i}'}
    for i in range(15)
]

# Define the layout of the dashboard
app.layout = html.Div(children=[
    html.H1("Anomaly Detection Dashboard", style={'font-family': 'Roboto, sans-serif', 'font-weight':'700', 'text-align': 'center'}),
    
    # Dropdown to select measurement
    html.Div(
    dcc.Dropdown(
        id='measurement-dropdown',
        options=measurement_options,
        value='Stage1.Output.Measurement0',  # Default selection
        placeholder="Select a measurement"
    ),
    style = {'width': '50%', 'margin': 'auto', 'padding':'10px'}
    ),

    # Graph to display the measurement trends and anomalies
    dcc.Graph(id='anomaly-graph'),

    # Heatmap for anomaly distribution
    dcc.Graph(id='downtime-bar-chart'),

    # Summary stats section
    html.Div(id='summary-stats', style={'font-family': 'Roboto, sans-serif', 'padding': '20px'}),
    ],
    style={'background-color': '#f9f9f9'}

    

)

# Callback function to update the summary stats based on the selected measurement
@app.callback(
        Output('summary-stats','children'),
        [Input('measurement-dropdown','value')]
)
def update_summary_stats(selected_measurement):
    # Filter data for the selected measurement (actual, setpoint, anomaly)
    setpoint_column = f'{selected_measurement}.U.Setpoint'
    
    # Extract the setpoint value
    setpoint_value = resampled_data_with_ano[setpoint_column].mean()

    # Calculate the upper and lower limits for the threshold (±10% of the setpoint)
    upper_limit = setpoint_value + (setpoint_value * 0.1)
    lower_limit = setpoint_value - (setpoint_value * 0.1)

    # Total anomalies
    anomaly_column = f'{selected_measurement}.U.Actual_anomaly'
    total_anomalies = resampled_data_with_ano[anomaly_column].sum()    

    return html.Div([
        html.H4(f'Summary for {selected_measurement}', style = {'text-align':'Left'}),
        html.P(f'Setpoint Value: {setpoint_value:.2f}', style={'font-weight': '500'}),
        html.P(f'Upper limit (+10%): {upper_limit:.2f}', style={'font-weight': '500'}),
        html.P(f'Lower Limit (-10%): {lower_limit:.2f}', style={'font-weight': '500'}),
        html.P(f'Total Anomalies Detected: {total_anomalies}', style={'font-weight': '500'})
    ])


# Callback function to update the graph based on dropdown selection
@app.callback(
    Output('anomaly-graph', 'figure'),
    [Input('measurement-dropdown', 'value')]
)
def update_anomaly_graph(selected_measurement):
    # Filter data for the selected measurement (actual, setpoint, anomaly)
    actual_column = f'{selected_measurement}.U.Actual'
    setpoint_column = f'{selected_measurement}.U.Setpoint'
    anomaly_column = f'{selected_measurement}.U.Actual_anomaly'

    # Create the figure
    fig = go.Figure()

    # Add the actual values trace
    fig.add_trace(go.Scatter(
        x=resampled_data_with_ano['time_stamp'], 
        y=resampled_data_with_ano[actual_column],
        mode='lines',
        name=f'{selected_measurement} Actual',
        line=dict(color='blue')
    ))

    # Add the setpoint values trace
    fig.add_trace(go.Scatter(
        x=resampled_data_with_ano['time_stamp'], 
        y=resampled_data_with_ano[setpoint_column],
        mode='lines',
        name=f'{selected_measurement} Setpoint',
        line=dict(color='orange')
    ))

    # Highlight anomaly points on the actual trend line
    fig.add_trace(go.Scatter(
        x=resampled_data_with_ano[resampled_data_with_ano[anomaly_column] == 1]['time_stamp'],  # Only anomaly points
        y=resampled_data_with_ano[resampled_data_with_ano[anomaly_column] == 1][actual_column],
        mode='markers',
        name=f'{selected_measurement} Anomalies',
        marker=dict(color='red', size=8, symbol='star')
    ))

    # Update the layout
    fig.update_layout(
        title=f'{selected_measurement} Actual vs Setpoint with Anomalies',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True
    )

    return fig

# Callback to update downtime bar chart
@app.callback(
    Output('downtime-bar-chart', 'figure'),
    [Input('measurement-dropdown', 'value')]
)
def update_downtime_bar_chart(selected_measurement):
    # Summarize downtime duration for both Stage 1 and Stage 2
    downtime_durations_stage1 = resampled_data_with_ano[resampled_data_with_ano['stage1_downtime_indicator'] == 1].groupby('downtime_event_stage1')['time_stamp'].agg(
        duration_seconds=('time_stamp', lambda x: (x.max() - x.min()).total_seconds())
    )
    downtime_durations_stage2 = resampled_data_with_ano[resampled_data_with_ano['stage2_downtime_indicator'] == 1].groupby('downtime_event_stage2')['time_stamp'].agg(
        duration_seconds=('time_stamp', lambda x: (x.max() - x.min()).total_seconds())
    )

    # Create a bar chart visualization for downtime durations
    fig = go.Figure()

    # Add bar trace for Stage 1
    fig.add_trace(go.Bar(
        x=downtime_durations_stage1.index,
        y=downtime_durations_stage1['duration_seconds'],
        name='Stage 1 Downtime Duration',
        marker_color='blue'
    ))

    # Add bar trace for Stage 2
    fig.add_trace(go.Bar(
        x=downtime_durations_stage2.index,
        y=downtime_durations_stage2['duration_seconds'],
        name='Stage 2 Downtime Duration',
        marker_color='orange'
    ))

    # Update layout
    fig.update_layout(
        title='Downtime Duration for Stage 1 and Stage 2',
        xaxis_title='Downtime Event',
        yaxis_title='Duration (seconds)',
        barmode='group'
    )

    return fig



 
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
