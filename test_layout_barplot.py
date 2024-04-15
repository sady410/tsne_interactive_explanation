import plotly.graph_objects as go

x = ["Author A", "Author B", "Author C", "Author D"]
y = [27, 18, 12, 6]
p = ['90%', '60%', '40%', '20%']

fig = go.Figure(
        go.Bar(
            x=[30]*4,
            y=x,
            text=p,
            textposition="auto",
            textfont=dict(color="white"),
            orientation="h",
            marker_color="grey",
        )
)

fig.add_trace(
    go.Bar(
        x=y,
        y=x,
        orientation="h",
    )
)
fig.update_layout(title='Top Authors', barmode="overlay", showlegend=False, template="presentation")
fig.update_yaxes(
    showticklabels=False,
)
fig.update_xaxes(range=[0, 30], visible=False)

for i in range(len(x)):
    fig.add_annotation(
        #x=y[i],
        y=x[i],
        text=x[i],
        showarrow=False,
        font=dict(color="black"),
        xshift=-600,
        yshift=65,
    )

fig.show()