import plotly.graph_objects as go

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [20, 35, 30, 25]

# Create horizontal bar plot
fig = go.Figure()

fig.add_trace(go.Bar(
    y=categories,
    x=values,
    orientation='h',
    text=values,  # Text to be displayed inside each bar
    textposition='auto',  # Position the text automatically below each bar
    textfont=dict(color='white'),  # Font color of the text
    marker=dict(color='rgb(0, 153, 255)')  # Color of the bars
))

# Customize layout
fig.update_layout(
    title='Horizontal Bar Plot with Labels Below Bars',
    xaxis_title='Values',
    yaxis_title='Categories',
    yaxis=dict(autorange='reversed')  # Reverse the y-axis to display categories from top to bottom
)

# Show plot
fig.show()
