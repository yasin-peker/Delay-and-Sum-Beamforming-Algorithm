import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Generate a sinusoidal signal with AWG noise
def generate_signal(sampling_rate=1e8,
                    number_of_samples=10000,
                    center_frequency=2e6):
    time = np.arange(number_of_samples) / sampling_rate

    # Create signal with AWG noise
    AWG_noise = np.random.normal(0, 0.1, number_of_samples) + 1j * np.random.normal(0, 0.1, number_of_samples)
    transmitter_signal = np.exp(2j * np.pi * center_frequency * time)
    transmitter_signal_with_noise = transmitter_signal + AWG_noise

    plt.figure()
    plt.title("Transmitter Signal with and without AWG Noise", fontsize=14, fontweight='bold')
    plt.plot(time * 1e6, transmitter_signal, label="TX Signal", color='b')
    plt.plot(time * 1e6, transmitter_signal_with_noise, label="TX Signal with AWG Noise", color='r')
    plt.legend(loc="upper right")
    plt.xlabel("Time [us]", fontsize=14, fontweight='bold')
    plt.ylabel("Voltage [V]", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    return time, AWG_noise, transmitter_signal, transmitter_signal_with_noise

# Delay-and-Sum Beamformer Implementation
def delay_and_sum_beamformer(time, AWG_noise, transmitter_signal, transmitter_signal_with_noise):
    "Construct the array steering vector"
    # Design the ULA with 3 element and half-wavelength spacing
    element_spacing = 0.5
    number_of_elements = 10

    # Define the direction of arrival of signals
    theta_degrees = 30
    # Convert the direction of arrival to radians
    theta = theta_degrees / 180 * np.pi

    # Compute the steering vector of the signal
    steering_vector = lambda theta, number_of_elements, element_spacing = element_spacing : np.exp(-2j * np.pi * element_spacing * np.arange(number_of_elements) * np.sin(theta))
    v_signal = steering_vector(theta, number_of_elements)[:, None]

    # To simulate the signal impinging on the array, we will matrix multiply the signal with the steering vector
    # and then add noise. In reality, each element will always be noisy and this is what we are simulating when
    # we add additional noise here which is denoted by σₑ.

    # X = s_tx * v_signal.T + σₑ
    # Where X is the impinging on the array
    AWG_noise_received = np.random.normal(loc=0, scale=0.1, size=(number_of_samples, number_of_elements)) + 1j * np.random.normal(loc=0, scale=0.1, size=(number_of_samples, number_of_elements))
    X = (transmitter_signal_with_noise[:, None] @ v_signal.T) + AWG_noise_received
    print(f"X.shape: {X.shape}")

    plt.figure()
    plt.title("Received Signal at Each Element", fontsize=14, fontweight='bold')
    for element_idx in range(number_of_elements):
        plt.plot(time * 1e6, X[:, element_idx], label=f"RX Signal Element {element_idx}")
    plt.legend(loc="upper right")
    plt.xlabel("Time [us]", fontsize=14, fontweight='bold')
    plt.ylabel("Voltage [V]", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    # Find the arrival angle
    # Define the angles to be searched
    thetas = np.arange(-90, 90 + 0.1, 0.05)

    # Define an output and response array to hold the results
    outputs = []
    responses = []

    # Iterate through each angle to find the angle of arrival
    for theta in thetas:
        theta *= np.pi/180

        # Steering vector
        w = steering_vector(theta, number_of_elements)

        # Process the signal at current angle
        y = (X @ w.conj()) / number_of_elements

        # Process the received signal with the steering vector
        # This processing is performed by matrix multiplying the Hermitian (Conjugate Transpose) of the
        # steering vector at angle v_s(theta) with the received signal matrix X, which produces an N dimensional vector y.

        # y = v_s(theta) * X

        # This matrix multiplication is where the time delays and signal summing occur, the steering vector actually
        # applies a time delay to the signal at each element. This is because there is a direct relation between time delay
        # and angle of arrival.

        # v_m(theta) = exp(-j * pi * m * sin(theta))

        # m * sin(theta) corresponds to the time delay based on physical location m of the element.
        # This time delay is proportional to angle theta

        # Compute array response
        # The array response is determined by the variance of the output vector y, if we have a small variance, then
        # we know that the received signals have a large degree of constructive interference and the output signal will be
        # the largest.

        # Compute power in dB
        array_response = 10 * np.log10(np.var(y))

        outputs.append(y)

        responses.append(array_response)

    # Normalize Array Responses
    responses -= np.max(responses)

    # Obtain angle that gives the maximum value
    angle_idx = np.argmax(responses)

    angle_of_arrival = thetas[angle_idx]

    print(f"Angle of arrival: {angle_of_arrival}")

    plt.figure()
    plt.title("Array Response", fontsize=14, fontweight='bold')
    plt.plot(thetas, responses, color='r')
    plt.xlabel("Angle [degrees]", fontsize=14, fontweight='bold')
    plt.ylabel("Response [dB]", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    # Plot the original and beamformed signals at the angle of arrival
    plt.figure()
    plt.title(f"Transmitted and Recevied Signal at Angle of Arrival {angle_of_arrival:.2f} Degrees", fontsize=14, fontweight='bold')
    plt.plot(time * 1e6, transmitter_signal_with_noise, color='b', label="TX Signal")
    plt.plot(time * 1e6, outputs[angle_idx], color='r', label="Beamformed RX Signal")
    plt.xlabel("Time [us]", fontsize=14, fontweight='bold')
    plt.ylabel("Voltage [V]", fontsize=14, fontweight='bold')
    plt.xlim(0, 10e-6)
    plt.legend(loc="upper right")
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    # Plot the received signal at different angle of arrival
    figure = plt.figure()

    # creating a plot with a red line and a label 'TX'
    lines_plotted = plt.plot([], [], color='red', label='TX')
    line_plotted = lines_plotted[0]

    # creating a plot with a blue line and a label 'RX'
    lines_plotted_2 = plt.plot([], [], color='blue', label='RX')
    line_plotted_2 = lines_plotted_2[0]

    plt.xlabel("Time [s]", fontsize=14, fontweight='bold')
    plt.ylabel("Voltage [V]", fontsize=14, fontweight='bold')
    plt.xlim(0, 400)
    plt.ylim(-1.25, 1.25)

    def animate(i):
        theta = thetas[i]  # Use your actual theta data
        y = outputs[i]  # Use your actual output data

        # update title
        plt.title(f"Received Signal Beamformed at Angle {theta:.2f} Degrees", fontsize=14, fontweight='bold')

        # update the data for both lines
        line_plotted.set_data(np.arange(0, 400), transmitter_signal_with_noise.real[:400])
        line_plotted_2.set_data(np.arange(0, 400), y.real[:400])

        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(loc="upper right")

    ani = animation.FuncAnimation(figure, animate, repeat=True, frames=np.arange(0, len(thetas), 3))

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=40, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('ds_signals_2.gif', writer=writer)


if __name__ == "__main__":

    print("##### Introduction to Beamforming #####")
    # In this code, the Delay-and-Sum Beamformer is implemented from scratch

    # Definition:
    # Beamforming is a signal processing technique that aims to estimate the Direction of signals
    # impinging on a sensor array.

    # Beamforming is achieved by combining elements such that some add constructively and others add
    # destructively.

    # In this algorithm, I considered a Uniform Linear Array (ULA) with M number of elements.

    # If the element spacing is known, then the distance that the wavefront must travel at each subsequent
    # element can be calculated.

    # The DAS beamformer applies a time delay to the incoming signal from each element and sums the output together.
    # If we get the time delays correct, we will have a single high output signal.
    # We can then use the time delays that produced this signal to determine the angle of it's arrival.

    # In order to calculate the time delays, we can steer the array acoss multiple angles and choose the angle
    # that produces the largest response.

    # We process the incoming signal with the steering vector to produce an array response, the angle that produces
    # the largest response, which is most likely the ange of arrival.
    sampling_rate = 1e8
    number_of_samples = 10000
    center_frequency = 2e6
    time, AWG_noise, transmitter_signal, transmitter_signal_with_noise = generate_signal(sampling_rate, number_of_samples, center_frequency)
    delay_and_sum_beamformer(time, AWG_noise, transmitter_signal, transmitter_signal_with_noise)