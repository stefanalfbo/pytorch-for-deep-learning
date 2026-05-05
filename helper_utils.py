import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    """
    Plots the actual data points and the model's predicted line for a given dataset.

    Args:
        model: The trained machine learning model to use for predictions.
        distances: The input data points (features) for the model.
        times: The target data points (labels) for the plot.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for efficient inference
    with torch.no_grad():
        # Make predictions using the trained model
        predicted_times = model(distances)

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the actual data points
    plt.plot(
        distances.numpy(),
        times.numpy(),
        color="orange",
        marker="o",
        linestyle="None",
        label="Actual Delivery Times",
    )

    # Plot the predicted line from the model
    plt.plot(
        distances.numpy(),
        predicted_times.numpy(),
        color="green",
        marker="None",
        label="Predicted Line",
    )

    # Set the title of the plot
    plt.title("Actual vs. Predicted Delivery Times")
    # Set the x-axis label
    plt.xlabel("Distance (miles)")
    # Set the y-axis label
    plt.ylabel("Time (minutes)")
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()


def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compares and plots the predictions of a model against new, non-linear data.

    Args:
        model: The trained model to be evaluated.
        new_distances: The new input data for generating predictions.
        new_times: The actual target values for comparison.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        # Generate predictions using the model
        predictions = model(new_distances)

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the actual data points
    plt.plot(
        new_distances.numpy(),
        new_times.numpy(),
        color="orange",
        marker="o",
        linestyle="None",
        label="Actual Data (Bikes & Cars)",
    )

    # Plot the predictions from the model
    plt.plot(
        new_distances.numpy(),
        predictions.numpy(),
        color="green",
        marker="None",
        label="Linear Model Predictions",
    )

    # Set the title of the plot
    plt.title("Linear Model vs. Non-Linear Reality")
    # Set the label for the x-axis
    plt.xlabel("Distance (miles)")
    # Set the label for the y-axis
    plt.ylabel("Time (minutes)")
    # Add a legend to the plot
    plt.legend()
    # Add a grid to the plot for better readability
    plt.grid(True)
    # Display the plot
    plt.show()


def plot_data(distances, times, normalize=False):
    """
    Creates a scatter plot of the data points.

    Args:
        distances: The input data points for the x-axis.
        times: The target data points for the y-axis.
        normalize: A boolean flag indicating whether the data is normalized.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))

    # Plot the data points as a scatter plot
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Delivery Times')

    # Check if the data is normalized to set appropriate labels and title
    if normalize:
        # Set the plot title for normalized data
        plt.title('Normalized Delivery Data (Bikes & Cars)')
        # Set the x-axis label for normalized data
        plt.xlabel('Normalized Distance')
        # Set the y-axis label for normalized data
        plt.ylabel('Normalized Time')
        # Display the legend
        plt.legend()
        # Add a grid to the plot
        plt.grid(True)
        # Show the plot
        plt.show()

    # Handle the case for un-normalized data
    else:
        # Set the plot title for un-normalized data
        plt.title('Delivery Data (Bikes & Cars)')
        # Set the x-axis label for un-normalized data
        plt.xlabel('Distance (miles)')
        # Set the y-axis label for un-normalized data
        plt.ylabel('Time (minutes)')
        # Display the legend
        plt.legend()
        # Add a grid to the plot
        plt.grid(True)
        # Show the plot
        plt.show()


def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    """
    Plots the predictions of a trained model against the original data,
    after de-normalizing the predictions.

    Args:
        model: The trained model used for prediction.
        distances: The original, un-normalized input data.
        times: The original, un-normalized target data.
        distances_norm: The normalized input data for the model.
        times_std: The standard deviation used for de-normalization.
        times_mean: The mean value used for de-normalization.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for prediction
    with torch.no_grad():
        # Get predictions from the model using normalized data
        predicted_norm = model(distances_norm)

    # De-normalize the predictions to their original scale
    predicted_times = (predicted_norm * times_std) + times_mean

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Data (Bikes & Cars)')

    # Plot the de-normalized predictions from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', label='Non-Linear Model Predictions')

    # Set the title of the plot
    plt.title('Non-Linear Model Fit vs. Actual Data')
    # Set the x-axis label
    plt.xlabel('Distance (miles)')
    # Set the y-axis label
    plt.ylabel('Time (minutes)')
    # Add a legend to the plot
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Display the plot
    plt.show()

    
def plot_training_progress(epoch, loss, model, distances_norm, times_norm, fig=None, ax=None):
    """
    Plots the training progress of a model on normalized data,
    showing the current fit at each epoch.

    Args:
        epoch: The current training epoch number.
        loss: The loss value at the current epoch.
        model: The model being trained.
        distances_norm: The normalized input data.
        times_norm: The normalized target data.
        fig: An optional Matplotlib figure object for plotting.
        ax: An optional Matplotlib axes object for plotting.
    """
    if fig is None or ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))

    predicted_norm = model(distances_norm)

    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    y_pred_plot = predicted_norm.detach().numpy()

    sorted_indices = x_plot.argsort(axis=0).flatten()

    ax.clear()
    ax.plot(
        x_plot,
        y_plot,
        color="orange",
        marker="o",
        linestyle="none",
        label="Actual Normalized Data",
    )
    ax.plot(
        x_plot[sorted_indices],
        y_pred_plot[sorted_indices],
        color="green",
        label="Model Predictions",
    )
    ax.set_title(f"Epoch: {epoch + 1} | Loss: {loss:.4f}")
    ax.set_xlabel("Normalized Distance")
    ax.set_ylabel("Normalized Time")
    ax.legend()
    ax.grid(True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)

    return fig, ax
