import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Set matplotlib style instead of seaborn
plt.style.use('ggplot')  # A clean, gridded style similar to seaborn's whitegrid

def load_sms_data(filepath):
    """
    Load SMS data from CSV file into a pandas DataFrame
    """
    try:
        # Read the CSV file with appropriate column headers
        # Extract column names from the first line of the file
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        # Parse column names from the first line
        column_names = []
        for col in first_line.split(','):
            # Extract the main parameter name without units
            name = col.split('{')[0].strip()
            column_names.append(name)
        
        # Read the CSV file, skipping the header line
        df = pd.read_csv(filepath, header=None, names=column_names, skiprows=1)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def explore_data(df):
    """
    Display basic information about the DataFrame
    """
    if df is None or df.empty:
        print("No data to explore")
        return
    
    print("\n=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Summary Statistics ===")
    print(df.describe())
    print("\n=== Missing Values ===")
    print(df.isna().sum())

def visualize_motion_data(df):
    """
    Create visualizations specifically for motion data
    """
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    # Create output directory for plots if it doesn't exist
    # output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(os.getcwd(), "plots")  # Use current working directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Extract time column for x-axis
        time_col = 'sys.exec.out.time' if 'sys.exec.out.time' in df.columns else 'sesmt.md.merged_frame.time'
        
        # Plot 1: Linear position over time
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz[0]'], label='X Position')
        plt.title('X Position Over Time')
        plt.ylabel('Position (m)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz[1]'], label='Y Position', color='orange')
        plt.title('Y Position Over Time')
        plt.ylabel('Position (m)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz[2]'], label='Z Position', color='green')
        plt.title('Z Position Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'linear_position_{timestamp}.png'))
        plt.close()
        
        # Plot 2: Linear velocity over time
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_vel[0]'], label='X Velocity')
        plt.title('X Velocity Over Time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_vel[1]'], label='Y Velocity', color='orange')
        plt.title('Y Velocity Over Time')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_vel[2]'], label='Z Velocity', color='green')
        plt.title('Z Velocity Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'linear_velocity_{timestamp}.png'))
        plt.close()
        
        # Plot 3: Linear acceleration over time
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_acc[0]'], label='X Acceleration')
        plt.title('X Acceleration Over Time')
        plt.ylabel('Acceleration (m/s²)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_acc[1]'], label='Y Acceleration', color='orange')
        plt.title('Y Acceleration Over Time')
        plt.ylabel('Acceleration (m/s²)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.xyz_acc[2]'], label='Z Acceleration', color='green')
        plt.title('Z Acceleration Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'linear_acceleration_{timestamp}.png'))
        plt.close()
        
        # Plot 4: Angular velocity over time
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.ang_vel[0]'], label='X Angular Velocity')
        plt.title('X Angular Velocity Over Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.ang_vel[1]'], label='Y Angular Velocity', color='orange')
        plt.title('Y Angular Velocity Over Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df[time_col], df['sesmt.md.merged_frame.ang_vel[2]'], label='Z Angular Velocity', color='green')
        plt.title('Z Angular Velocity Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'angular_velocity_{timestamp}.png'))
        plt.close()
        
        # Plot 5: 3D trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['sesmt.md.merged_frame.xyz[0]'], 
                df['sesmt.md.merged_frame.xyz[1]'], 
                df['sesmt.md.merged_frame.xyz[2]'])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')  # type: ignore
        ax.set_title('3D Trajectory')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'3d_trajectory_{timestamp}.png'))
        plt.close()
        
        # Plot 6: Quaternion components over time
        plt.figure(figsize=(12, 8))
        plt.plot(df[time_col], df['sesmt.md.merged_frame.quat[0]'], label='q0')
        plt.plot(df[time_col], df['sesmt.md.merged_frame.quat[1]'], label='q1')
        plt.plot(df[time_col], df['sesmt.md.merged_frame.quat[2]'], label='q2')
        plt.plot(df[time_col], df['sesmt.md.merged_frame.quat[3]'], label='q3')
        plt.title('Quaternion Components Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'quaternion_{timestamp}.png'))
        plt.close()
        
        # Plot 7: Gravity components over time
        plt.figure(figsize=(12, 8))
        plt.plot(df[time_col], df['sesmt.md.merged_frame.gravity[0]'], label='X Gravity')
        plt.plot(df[time_col], df['sesmt.md.merged_frame.gravity[1]'], label='Y Gravity')
        plt.plot(df[time_col], df['sesmt.md.merged_frame.gravity[2]'], label='Z Gravity')
        plt.title('Gravity Components Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Gravity (m/s²)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gravity_{timestamp}.png'))
        plt.close()
        
        print(f"Motion data visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error creating motion data visualizations: {str(e)}")

def compute_average_acceleration(df, start_time=40, end_time=90):
    """
    Compute average linear accelerations over a specified time interval
    
    Args:
        df: DataFrame containing SMS data
        start_time: Start time in seconds (default: 40)
        end_time: End time in seconds (default: 90)
        
    Returns:
        Dictionary containing average acceleration values for each axis
    """
    if df is None or df.empty:
        print("No data to compute accelerations")
        return None
    
    try:
        # Determine the time column
        time_col = 'sys.exec.out.time' if 'sys.exec.out.time' in df.columns else 'sesmt.md.merged_frame.time'
        
        # Filter data within the specified time range
        filtered_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
        
        if filtered_df.empty:
            print(f"No data found within the time range {start_time}s to {end_time}s")
            return None
        
        # Compute average accelerations for each axis
        avg_acc_x = filtered_df['sesmt.md.merged_frame.xyz_acc[0]'].mean()
        avg_acc_y = filtered_df['sesmt.md.merged_frame.xyz_acc[1]'].mean()
        avg_acc_z = filtered_df['sesmt.md.merged_frame.xyz_acc[2]'].mean()
        
        # Compute magnitude of average acceleration
        avg_acc_magnitude = np.sqrt(avg_acc_x**2 + avg_acc_y**2 + avg_acc_z**2)
        
        # Store results in a dictionary
        results = {
            'x_axis': avg_acc_x,
            'y_axis': avg_acc_y,
            'z_axis': avg_acc_z,
            'magnitude': avg_acc_magnitude
        }
        
        print("\n=== Average Linear Accelerations (40s to 90s) ===")
        print(f"X-axis: {avg_acc_x:.4f} m/s²")
        print(f"Y-axis: {avg_acc_y:.4f} m/s²")
        print(f"Z-axis: {avg_acc_z:.4f} m/s²")
        print(f"Magnitude: {avg_acc_magnitude:.4f} m/s²")
        
        return results
    
    except Exception as e:
        print(f"Error computing average accelerations: {str(e)}")
        return None

def main():
    # Path to the SMS data file
    file_path = "/Users/jozbee/work/eng/comp/data/00_sms_drive.csv"
    
    # Load the data
    sms_df = load_sms_data(file_path)
    
    # Explore the data
    explore_data(sms_df)
    
    # Compute average accelerations between 40s and 90s
    compute_average_acceleration(sms_df, 40, 90)
    
    # Visualize motion data
    visualize_motion_data(sms_df)

if __name__ == "__main__":
    main()
