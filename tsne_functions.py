import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _Hbeta(D = np.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = np.maximum(sum(P), np.finfo('double').eps)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P
    

def _x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax =  np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = _Hbeta(Di, beta[i])
            
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while (np.isnan(Hdiff) or np.abs(Hdiff) > tol) and tries < 50:
                
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            (H, thisP) = _Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1
            
        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    
    return P, np.sqrt(1 / beta)


def compute_tsne(X, no_dims = 2, perplexity = 30.0, max_iter = 400, Y_init = None):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if X.dtype != "float64":
        print("Error: array X should have type float64.")
        return -1
    #if no_dims.__class__ != "<type 'int'>":			# doesn't work yet!
    #	print "Error: number of dimensions should be an integer.";
    #	return -1;

    (n, d) = X.shape

    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    if not isinstance(Y_init, np.ndarray):
        Y = np.random.randn(n, no_dims)
    else:
        Y = Y_init.copy()
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P, sigma = _x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4 # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
            
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
            
        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    return Y, P, Q, sigma

###########################################
#############  PLOTTING  ##################
###########################################

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
        font=dict(size=5),
        xaxis=dict(tickfont=dict(size=5), ticks="outside", showgrid=False, zeroline=False, mirror=True),
        yaxis=dict(tickfont=dict(size=5), ticks="outside", showgrid=False, zeroline=False, mirror=True),
        margin=dict(l=2, r=2, t=2, b=2),
        template="simple_white"
    )

    fig.update_traces(marker={"size": 10, "line":dict(width=2, color="DarkSlateGrey")})
    
    return fig