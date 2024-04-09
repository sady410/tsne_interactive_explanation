import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import hmean
from sklearn.metrics import pairwise_distances


def compute_all_gradients(X, Y, P, Q, sigma):
    gradients = []
    for i in range(X.shape[0]):
        gradients.append(compute_gradients(X, Y, P, Q, sigma, i))
    
    gradients = np.array(gradients)
    
    return gradients

def compute_gradients(X, Y, P, Q, sigma, i):
    """
    Function that compute the saliency for an image X and output y.

    Parameters:
    -----------
    i: indice of the input to consider
    
    Return:
    -------
    derivative: t-sne "saliency"
    """
    y2_derivative = _compute_y2_derivative(i, Y, P, Q)
    
    yx_derivative = _compute_xy_derivative(i, X, Y, sigma)
    
    derivative = (-np.linalg.inv(y2_derivative)) @ (yx_derivative.T)

    return derivative

def _compute_y2_derivative(i, y, P, Q):
    """
    Function that compute the second derivative of t-sne regarding y_i
    
    Parameters:
    -----------
    i: indice of the input to consider
    y: low-dimensional space embedding
    P: p-values of t-sne
    Q: q-values of t-sne
    
    Return:
    -------
    4*res: derivative regarding y_i
    """
    n = y.shape[0]
    m = y.shape[1]

    d_ij = y[i] - y
    d_ij_d = np.identity(m, np.float64)
    d_ij_d = np.tile(d_ij_d, (n, 1, 1))

    e_ij = 1 + (np.linalg.norm(d_ij, axis=1))**2
    e_ij_d = 2 * d_ij
    E_ij = 1/e_ij

    distances = 1 + pairwise_distances(y, squared=True) # refactor e_ij with this maybe
    S_q = np.sum( 1/distances ) - np.trace(1/distances)
    S_q_d = -4 * np.sum((e_ij**(-2)).reshape(1, n) @ d_ij, axis=0)
    
    v_ij = (P - Q)[i]     
    v_ij_d = ( ( e_ij_d.T * e_ij**(-2) * S_q ) + ( S_q_d.reshape(m, 1) / e_ij.reshape(1, n) ) ) / ( S_q**2 )
    
    term1 = (v_ij_d * E_ij.reshape(1, n)) @ d_ij
    term2 = (e_ij_d.T * (v_ij * E_ij**2).reshape(1, n)) @ d_ij
    term3 = np.delete( ((v_ij * E_ij).reshape(n, 1, 1) * d_ij_d), i, axis=0).sum(axis=0)
    
    return 4 * ( term1 - term2 + term3 ) 
    
def _compute_xy_derivative(i, X, y, sigma):
    """
    Function that compute the second derivative of t-sne regarding x_i

    Parameters:
    -----------
    i: indice of the input to consider
    X: instances in high-dimensional space
    y: embedding in low-dimensional space 
    sigma: sigma values found by t-sne with the chosen perplexity 
    
    Return:
    -------
    4*res: derivative regarding x_i
    """
    n = y.shape[0]
    m = y.shape[1]
    
    sigma = sigma.reshape((X.shape[0],))
    
    y_ij = y[i] - y
    x_ij = X[i] - X
    x_ji = X - X[i]
    e_ij = 1 + (np.linalg.norm(y_ij, axis=1))**2
    E_ij = 1/e_ij

    exp_ij = np.exp( -( (np.linalg.norm(x_ij, axis=1)**2) / (2*(sigma[i]**2)) ) )
    exp_ji = np.exp( -( (np.linalg.norm(x_ji, axis=1)**2) / (2*(sigma**2) ) ))

    S_pi = np.delete( exp_ij, i ).sum()

    S_pi_d = (- (x_ij) * (sigma[i]**(-2)) * exp_ij.reshape(n, 1)).sum(axis=0)
    S_pi_d = np.tile(S_pi_d , (n, 1))
    
    S_pj = ((np.exp( - pairwise_distances(X, squared=True) / ( 2*(sigma**2) ) )).sum(axis=0)) - 1 # on enlève la diagonale (quand j = l)
    S_pj_d = (sigma**(-2) * exp_ji).reshape(n,1) * x_ji

    P_ji_d = ( ( -S_pi * x_ij * sigma[i]**(-2) * exp_ij.reshape(n, 1)) - ( exp_ij.reshape(n, 1) * S_pi_d ) ) / S_pi**2
    P_ij_d = ( ( (S_pj * sigma**(-2) * exp_ji).reshape(n,1) * x_ji ) - ( exp_ji.reshape(n,1) * S_pj_d ) ) / (S_pj**2).reshape(n,1)
    
    v_ij_d = (1 / (2*n)) * (P_ji_d + P_ij_d)

    # return 4 * np.delete( ((v_ij_d * E_ij).reshape(n, 1, 1) * y_ij), i, axis=0).sum(axis=0)
    return 4 * ( v_ij_d.T @ ( y_ij * E_ij.reshape(n, 1) ) )


###################################
##########  PLOTTING  #############
###################################


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

def create_arrow_fields_plot(scaled_gradients, features, feature_id, scale = 1):

    fig = ff.create_quiver(Y[:, 0], Y[:, 1], scaled_gradients[:, 0, feature_id], scaled_gradients[:, 1, feature_id], scale=scale)

    # fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=np.array(activations[:, i])))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, mirror=True),
        yaxis=dict(showgrid=False, zeroline=False, mirror=True),
        plot_bgcolor= "rgba(0, 0, 0, 0)",
        paper_bgcolor= "rgba(0, 0, 0, 0)",
        showlegend=False,
        template="simple_white"
    )

    return fig

def combined_gradients_plot(gradients, features, instance_id):
    combined_magnitude = np.linalg.norm(gradients[instance_id], axis=0)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=combined_magnitude, y=features, orientation="h", name="Combined Gradients", showlegend=False))

    fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, zerolinecolor="grey", zerolinewidth=1, mirror=True)

    fig.update_layout(
        height=1000, width=900, font=dict(size=15), template="simple_white",
        plot_bgcolor= "rgba(0, 0, 0, 0)",
        paper_bgcolor= "rgba(0, 0, 0, 0)",
    )
    
    return fig

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