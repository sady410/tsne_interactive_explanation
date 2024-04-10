
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


################################
######### Other plots ##########
################################

def create_average_feature_distribution_plot(feature_names, X):
    pass

################################
####### t-SNE plots ############
################################

def create_plot_tsne_embedding(X, Y, targets):

    df = pd.DataFrame()
    df["id"] = np.array([i for i in range(X.shape[0])])
    df["class"] = np.array([str(i) for i in targets])
    df["comp-1"] = Y[:,0]
    df["comp-2"] = Y[:,1]

    fig = px.scatter(df, x="comp-1", y="comp-2", color="class", hover_data=["id"])
    fig.update_layout({
        'plot_bgcolor': "rgba(0, 0, 0, 0)",
        'paper_bgcolor': "rgba(0, 0, 0, 0)",
    })

    fig.update_layout(
        showlegend=False,
        yaxis_title=None,
        xaxis_title=None,
        xaxis=dict(showticklabels=False, mirror=True),
        yaxis=dict(showticklabels=False, mirror=True),
        margin=dict(l=2, r=2, t=2, b=2),
        template="simple_white"
    )

    fig.update_traces(marker={"size": 10, "line":dict(width=2, color="DarkSlateGrey")})
    
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

    fig.add_trace(go.Bar(x=mean_norms, y=features, orientation="h", showlegend=False))

    fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, zerolinecolor="grey", zerolinewidth=1, mirror=True)

    fig.update_layout( 
            template="simple_white",
            plot_bgcolor= "rgba(0, 0, 0, 0)",
            paper_bgcolor= "rgba(0, 0, 0, 0)",
    )

    return fig

def create_combined_gradients_plot(gradients, features, instance_idx):
    combined_magnitude = np.linalg.norm(gradients[instance_idx], axis=0)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=combined_magnitude, y=features, orientation="h", name="Combined Gradients", showlegend=False))

    fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, zerolinecolor="grey", zerolinewidth=1, mirror=True)

    fig.update_layout( 
            template="simple_white",
            plot_bgcolor= "rgba(0, 0, 0, 0)",
            paper_bgcolor= "rgba(0, 0, 0, 0)",
    )
    
    return fig

# def create_arrow_fields_plot(Y, scaled_gradients, features, feature_id, scale = 1):

#     fig = ff.create_quiver(Y[:, 0], Y[:, 1], scaled_gradients[:, 0, feature_id], scaled_gradients[:, 1, feature_id], scale=scale)

#     # fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=np.array(activations[:, i])))

#     fig.update_layout(
#         xaxis=dict(showgrid=False, zeroline=False, mirror=True),
#         yaxis=dict(showgrid=False, zeroline=False, mirror=True),
#         plot_bgcolor= "rgba(0, 0, 0, 0)",
#         paper_bgcolor= "rgba(0, 0, 0, 0)",
#         showlegend=False,
#         template="simple_white"
#     )

#     return fig


def create_top_gradient_vectors_plot(gradients, Y, features, instance_id, nb_features):

    combined_magnitude = np.linalg.norm(gradients[instance_id], axis=0)
    top_features_indices = np.argsort(combined_magnitude)[-nb_features:]
    vectors = gradients[instance_id][:, top_features_indices]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Y[:, 0],
        y=Y[:, 1],
        mode="markers",
        marker=dict(size=7),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[Y[instance_id, 0]],
        y=[Y[instance_id, 1]],
        mode="markers",
        marker=dict(color="crimson", size=7),
        name="Instance " + str(instance_id),
    ))

    # Unpack vectors for plotting
    x_endpoints = [Y[instance_id, 0] + vector[0]*20 for vector in vectors.T]
    y_endpoints = [Y[instance_id, 1] + vector[1]*20 for vector in vectors.T]

    colors = ['yellow', 'green', 'blue', 'orange', 'purple']  # Choose different colors for each line

    for i, (x_end, y_end, feature_id) in enumerate(zip(x_endpoints, y_endpoints, top_features_indices)):
        fig.add_trace(go.Scatter(
            x=[Y[instance_id, 0], x_end],
            y=[Y[instance_id, 1], y_end],
            mode="lines",
            line=dict(color=colors[i], width=2),
            name=features[feature_id],
        ))

    fig.update_layout(
        template="simple_white",
        plot_bgcolor= "rgba(0, 0, 0, 0)",
        paper_bgcolor= "rgba(0, 0, 0, 0)",
        # legend=dict(
        #     x=1,
        #     y=1,
        #     xanchor='right',
        #     yanchor='top',
        #     bgcolor='rgba(255, 255, 255, 0)',
        #     bordercolor='rgba(255, 255, 255, 0)'
        # )
    )

    return fig