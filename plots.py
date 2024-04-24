
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from css_colors_exposed import Color

################################
######### Other plots ##########
################################


def create_average_feature_distribution_plot(features, X, idx):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.mean(X[idx], axis=0), y=features,
                  orientation="h", name="Combined Gradients", showlegend=False))
    
    fig.update_yaxes(showticklabels=False, showline=True, ticks="")
    fig.update_xaxes(ticks="outside", showline=True, showgrid=False)


    fig.update_layout(
        template="simple_white",
        plot_bgcolor=Color.transparent.value,
        paper_bgcolor=Color.transparent.value,
        margin=dict(l=0,r=0,b=0,t=0)
    )
    fig.update_traces(marker={
        "color": [Color.primaryBorderSubtle.value for _ in range(len(features))],
        "line": dict(width=3, color=[Color.primaryBorderSubtle.value for _ in range(len(features))]),
    })

    return fig

################################
####### t-SNE plots ############
################################


def create_plot_tsne_embedding(X, Y, targets, dataset_name):

    df = pd.DataFrame()
    df["Id"] = np.array([i for i in range(X.shape[0])])
    df["Comp-1"] = Y[:, 0]
    df["Comp-2"] = Y[:, 1]


    if dataset_name == "countries":
        df["Country"] = targets
        fig = px.scatter(df, x="Comp-1", y="Comp-2",
                         hover_name="Country", hover_data=["Id"])
    elif dataset_name == "diabetes":
        df["Disease Measure"] = targets
        fig = px.scatter(df, x="Comp-1", y="Comp-2",
                         hover_name="Disease Measure", hover_data=["Id"])
    elif dataset_name == "zoo":
        df["Animal"] = targets
        fig = px.scatter(df, x="Comp-1", y="Comp-2",
                         hover_name="Animal", hover_data=["Id"])
    else:
        df["Class"] = np.array([str(i) for i in targets])
        df["Species"] = np.array([["Setosa", "Versicolor", "Virginica"][i] for i in targets])
        fig = px.scatter(df, x="Comp-1", y="Comp-2",
                        color="Class", hover_name="Species", hover_data=["Id"])
    
    fig.update_layout(
        showlegend=False,
        yaxis_title=None,
        xaxis_title=None,
        xaxis_visible=False, 
        yaxis_visible=False,
        margin=dict(l=0.5, r=0.5, t=0.5, b=0.5),
        template="simple_white",
        plot_bgcolor=Color.transparent.value,
        paper_bgcolor=Color.transparent.value,
        hoverlabel=dict(
            bgcolor=Color.secondary.value,
            font_size=16,     
        ) 
    )

    fig.update_traces(marker={
        "size": 10, 
        "line": dict(width=2, color=Color.gray700.value),
    })

    if dataset_name != "iris":
        
        fig.update_traces(marker={
            "color": Color.primary.value
        })

    return fig


################################
####### Explainer plots ########
################################


def create_feature_importance_ranking_plot(gradients, features):

    norms = np.zeros(gradients.shape[2])

    for g in gradients:
        norm = np.linalg.norm(g, axis=0)
        norms += (norm / np.sum(norm))

    mean_norms = norms/gradients.shape[2]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=mean_norms, y=features, textposition="auto", text=features, 
                  orientation="h", showlegend=False))

    fig.update_yaxes(showticklabels=False, showline=True, ticks="")
    fig.update_xaxes(ticks="outside", showline=True, showgrid=False)

    fig.update_layout(
        template="simple_white",
        plot_bgcolor=Color.transparent.value,
        paper_bgcolor=Color.transparent.value,
        margin=dict(l=0,r=0,b=0,t=0),
        height=len(features) * 50,
    )

    fig.update_traces(marker={
        "color": [Color.primaryBorderSubtle.value for _ in range(len(features))]
    })

    return fig


def create_combined_gradients_plot(gradients, features, instance_idx):
    combined_magnitude = np.linalg.norm(gradients[instance_idx], axis=0)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=combined_magnitude, y=features, textposition="auto", text=features, 
                  orientation="h", name="Combined Gradients", showlegend=False))

    fig.update_yaxes(showticklabels=False, showline=True, ticks="")
    fig.update_xaxes(ticks="outside", showline=True, showgrid=False)

    fig.update_layout(
        template="simple_white",
        plot_bgcolor=Color.transparent.value,
        paper_bgcolor=Color.transparent.value,
         margin=dict(l=0,r=0,b=0,t=0)
    )
    
    fig.update_traces(marker={
        "color": [Color.primaryBorderSubtle.value for _ in range(len(features))]
    })

    return fig
