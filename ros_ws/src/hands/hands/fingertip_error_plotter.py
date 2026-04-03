#!/usr/bin/env python3
"""
Fingertip Error Plotter

Subscribes to /fingertip_errors and plots x, y, z error vs time for each finger.
Error = target position (Rokoko) - actual position (PyBullet FK)

Usage:
    ros2 run hands fingertip_error_plotter
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Ctrl+C handling
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import signal
import sys

class FingertipErrorPlotter(Node):
    def __init__(self):
        super().__init__('fingertip_error_plotter')

        # Subscribe to fingertip errors
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'fingertip_errors',
            self.error_callback,
            10)

        # Data storage - store ALL data for saving (no max limit)
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.axis_names = ['x', 'y', 'z']

        # Full data storage (used for both live display and saving)
        self.all_time_data = []
        self.all_error_data = {
            finger: {axis: [] for axis in self.axis_names}
            for finger in self.finger_names
        }
        # Total distance error storage (Euclidean distance)
        self.all_total_error = {finger: [] for finger in self.finger_names}

        # Output directory for saved plots
        self.output_dir = os.path.expanduser('~/Desktop/hand_control_ws/error_plots')
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f'Plots will be saved to: {self.output_dir}')

        # Set up matplotlib for interactive plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        self.fig.suptitle('Fingertip Position Error (Target - Actual) vs Time', fontsize=14)

        # Colors for x, y, z
        self.colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

        # Initialize line objects for each subplot
        self.lines = {}
        for i, finger in enumerate(self.finger_names):
            self.axes[i].set_ylabel(f'{finger.capitalize()}\nError (m)', fontsize=10)
            self.axes[i].grid(True, alpha=0.3)
            self.axes[i].set_ylim(-0.05, 0.05)  # ±5cm initial range
            self.lines[finger] = {}
            for axis in self.axis_names:
                line, = self.axes[i].plot([], [], color=self.colors[axis],
                                          label=f'{axis}', linewidth=1)
                self.lines[finger][axis] = line
            self.axes[i].legend(loc='upper right', fontsize=8)

        self.axes[-1].set_xlabel('Time (s)', fontsize=12)

        plt.tight_layout()
        plt.show(block=False)

        # Second figure: Total distance error (Euclidean)
        self.fig2, self.axes2 = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        self.fig2.suptitle('Total Distance Error (Euclidean) vs Time', fontsize=14)

        # Colors for each finger
        self.finger_colors = {
            'thumb': 'red',
            'index': 'green',
            'middle': 'blue',
            'ring': 'orange',
            'pinky': 'purple'
        }

        # Initialize line objects for total error plot
        self.lines2 = {}
        for i, finger in enumerate(self.finger_names):
            self.axes2[i].set_ylabel(f'{finger.capitalize()}\nError (m)', fontsize=10)
            self.axes2[i].grid(True, alpha=0.3)
            self.axes2[i].set_ylim(0, 0.1)  # 0-10cm initial range (distance is always positive)
            line, = self.axes2[i].plot([], [], color=self.finger_colors[finger],
                                       label=f'Total', linewidth=1)
            self.lines2[finger] = line
            self.axes2[i].legend(loc='upper right', fontsize=8)

        self.axes2[-1].set_xlabel('Time (s)', fontsize=12)

        plt.tight_layout()
        plt.show(block=False)

        # Timer for updating plot
        self.create_timer(0.1, self.update_plot)  # Update at 10 Hz

        # Track last data received time for auto-save on playback end
        self.last_data_time = None
        self.data_timeout_sec = 1.0  # Save and exit if no data for 1 second
        self.has_received_data = False

        self.get_logger().info('Fingertip Error Plotter started. Waiting for data on /fingertip_errors...')
        self.get_logger().info(f'Will auto-save if no data received for {self.data_timeout_sec} seconds after playback starts.')

    def error_callback(self, msg):
        """Process incoming error data"""
        data = msg.data

        if len(data) < 16:  # 1 time + 5 fingers * 3 axes = 16
            return

        # Log first data received
        if not hasattr(self, '_first_data_logged'):
            self._first_data_logged = True
            self.get_logger().info(f'Receiving data! First message has {len(data)} values')

        # Track last data time for auto-save
        self.last_data_time = self.get_clock().now()
        self.has_received_data = True

        # Extract time
        time = data[0]
        self.all_time_data.append(time)

        # Periodic logging
        if len(self.all_time_data) % 100 == 0:
            self.get_logger().info(f'Collected {len(self.all_time_data)} data points')

        # Extract errors for each finger (3 values each: x, y, z)
        for i, finger in enumerate(self.finger_names):
            idx = 1 + i * 3  # Start index for this finger
            ex, ey, ez = data[idx], data[idx + 1], data[idx + 2]
            self.all_error_data[finger]['x'].append(ex)
            self.all_error_data[finger]['y'].append(ey)
            self.all_error_data[finger]['z'].append(ez)
            # Calculate total Euclidean distance error
            total_error = np.sqrt(ex**2 + ey**2 + ez**2)
            self.all_total_error[finger].append(total_error)

    def update_plot(self):
        """Update the plot with all data (no sliding window)"""
        # Check for data timeout (playback ended)
        if self.has_received_data and self.last_data_time is not None:
            time_since_last = (self.get_clock().now() - self.last_data_time).nanoseconds / 1e9
            if time_since_last > self.data_timeout_sec:
                self.get_logger().info(f'No data for {time_since_last:.1f}s - playback ended. Auto-saving...')
                self.auto_save_and_exit()
                return

        if len(self.all_time_data) < 2:
            return

        time_array = np.array(self.all_time_data)

        for i, finger in enumerate(self.finger_names):
            for axis in self.axis_names:
                error_array = np.array(self.all_error_data[finger][axis])
                if len(error_array) == len(time_array):
                    self.lines[finger][axis].set_data(time_array, error_array)

            # Auto-scale y-axis based on data
            all_errors = []
            for axis in self.axis_names:
                err = np.array(self.all_error_data[finger][axis])
                valid = err[~np.isnan(err)]
                if len(valid) > 0:
                    all_errors.extend(valid)

            if len(all_errors) > 0:
                min_err = min(all_errors)
                max_err = max(all_errors)
                margin = max(0.01, (max_err - min_err) * 0.1)
                self.axes[i].set_ylim(min_err - margin, max_err + margin)

            # Update x-axis limits
            self.axes[i].set_xlim(time_array[0], time_array[-1])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Update second figure (total distance error)
        for i, finger in enumerate(self.finger_names):
            total_err_array = np.array(self.all_total_error[finger])
            if len(total_err_array) == len(time_array):
                self.lines2[finger].set_data(time_array, total_err_array)

            # Auto-scale y-axis based on data
            valid = total_err_array[~np.isnan(total_err_array)]
            if len(valid) > 0:
                max_err = max(valid)
                margin = max(0.01, max_err * 0.1)
                self.axes2[i].set_ylim(0, max_err + margin)

            # Update x-axis limits
            self.axes2[i].set_xlim(time_array[0], time_array[-1])

        self.fig2.canvas.draw_idle()
        self.fig2.canvas.flush_events()

    def auto_save_and_exit(self):
        """Auto-save plots and exit when playback ends"""
        self.get_logger().info('Auto-saving plots and exiting...')
        try:
            plt.ioff()
            plt.close('all')
            print(f'Collected {len(self.all_time_data)} data points')
            self.save_plots()
            print('Done! Exiting...')
        except Exception as e:
            print(f'Error saving: {e}')
            import traceback
            traceback.print_exc()

        # Exit the program
        os._exit(0)

    def save_plots(self):
        """Save the complete error plots as images"""
        self.get_logger().info(f'save_plots called with {len(self.all_time_data)} data points')

        if len(self.all_time_data) < 2:
            self.get_logger().warn('Not enough data to save plots (need at least 2 points)')
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            time_array = np.array(self.all_time_data)
            self.get_logger().info(f'Saving plots with timestamp {timestamp}...')

            # Create a new figure for the full data
            fig_full, axes_full = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
            fig_full.suptitle(f'Fingertip Position Error (Target - Actual) vs Time\n{timestamp}', fontsize=14)

            for i, finger in enumerate(self.finger_names):
                axes_full[i].set_ylabel(f'{finger.capitalize()}\nError (m)', fontsize=10)
                axes_full[i].grid(True, alpha=0.3)

                for axis in self.axis_names:
                    error_array = np.array(self.all_error_data[finger][axis])
                    if len(error_array) == len(time_array):
                        axes_full[i].plot(time_array, error_array, color=self.colors[axis],
                                          label=f'{axis}', linewidth=0.8)

                axes_full[i].legend(loc='upper right', fontsize=8)

                # Auto-scale y-axis
                all_errors = []
                for axis in self.axis_names:
                    err = np.array(self.all_error_data[finger][axis])
                    valid = err[~np.isnan(err)]
                    if len(valid) > 0:
                        all_errors.extend(valid)

                if len(all_errors) > 0:
                    min_err = min(all_errors)
                    max_err = max(all_errors)
                    margin = max(0.005, (max_err - min_err) * 0.1)
                    axes_full[i].set_ylim(min_err - margin, max_err + margin)

            axes_full[-1].set_xlabel('Time (s)', fontsize=12)
            plt.tight_layout()

            # Save combined plot
            combined_path = os.path.join(self.output_dir, f'fingertip_errors_{timestamp}.png')
            fig_full.savefig(combined_path, dpi=150, bbox_inches='tight')
            self.get_logger().info(f'Saved combined plot: {combined_path}')

            # Save individual finger plots
            for i, finger in enumerate(self.finger_names):
                fig_single, ax_single = plt.subplots(figsize=(12, 4))
                ax_single.set_title(f'{finger.capitalize()} Finger Error vs Time - {timestamp}', fontsize=12)
                ax_single.set_xlabel('Time (s)', fontsize=10)
                ax_single.set_ylabel('Error (m)', fontsize=10)
                ax_single.grid(True, alpha=0.3)

                for axis in self.axis_names:
                    error_array = np.array(self.all_error_data[finger][axis])
                    if len(error_array) == len(time_array):
                        ax_single.plot(time_array, error_array, color=self.colors[axis],
                                       label=f'{axis}', linewidth=0.8)

                ax_single.legend(loc='upper right', fontsize=10)
                plt.tight_layout()

                finger_path = os.path.join(self.output_dir, f'{finger}_error_{timestamp}.png')
                fig_single.savefig(finger_path, dpi=150, bbox_inches='tight')
                plt.close(fig_single)

            plt.close(fig_full)
            self.get_logger().info(f'Saved {len(self.finger_names)} individual finger plots to {self.output_dir}')

            # Save total distance error plot (combined)
            fig_total, axes_total = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
            fig_total.suptitle(f'Total Distance Error (Euclidean) vs Time\n{timestamp}', fontsize=14)

            for i, finger in enumerate(self.finger_names):
                axes_total[i].set_ylabel(f'{finger.capitalize()}\nError (m)', fontsize=10)
                axes_total[i].grid(True, alpha=0.3)

                total_err_array = np.array(self.all_total_error[finger])
                if len(total_err_array) == len(time_array):
                    axes_total[i].plot(time_array, total_err_array, color=self.finger_colors[finger],
                                       label='Total', linewidth=0.8)

                axes_total[i].legend(loc='upper right', fontsize=8)

                # Auto-scale y-axis
                valid = total_err_array[~np.isnan(total_err_array)]
                if len(valid) > 0:
                    max_err = max(valid)
                    margin = max(0.005, max_err * 0.1)
                    axes_total[i].set_ylim(0, max_err + margin)

            axes_total[-1].set_xlabel('Time (s)', fontsize=12)
            plt.tight_layout()

            total_plot_path = os.path.join(self.output_dir, f'total_distance_errors_{timestamp}.png')
            fig_total.savefig(total_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_total)
            self.get_logger().info(f'Saved total distance error plot: {total_plot_path}')

            # Also save raw data as CSV (including total error)
            csv_path = os.path.join(self.output_dir, f'fingertip_errors_{timestamp}.csv')
            with open(csv_path, 'w') as f:
                # Header
                header = ['time']
                for finger in self.finger_names:
                    for axis in self.axis_names:
                        header.append(f'{finger}_{axis}')
                    header.append(f'{finger}_total')  # Add total error column
                f.write(','.join(header) + '\n')

                # Data rows
                for t_idx in range(len(self.all_time_data)):
                    row = [str(self.all_time_data[t_idx])]
                    for finger in self.finger_names:
                        for axis in self.axis_names:
                            row.append(str(self.all_error_data[finger][axis][t_idx]))
                        row.append(str(self.all_total_error[finger][t_idx]))  # Add total error
                    f.write(','.join(row) + '\n')

            self.get_logger().info(f'Saved raw data: {csv_path}')

            # Calculate and save error statistics
            stats_path = os.path.join(self.output_dir, f'error_statistics_{timestamp}.txt')
            stats_csv_path = os.path.join(self.output_dir, f'error_statistics_{timestamp}.csv')

            print('\n' + '='*70)
            print('ERROR STATISTICS SUMMARY')
            print('='*70)

            # Prepare CSV data
            csv_rows = []
            csv_rows.append(['finger', 'axis', 'RMSE (mm)', 'MAE (mm)', 'Max Error (mm)', 'Total Error (mm)'])

            with open(stats_path, 'w') as f:
                f.write('ERROR STATISTICS SUMMARY\n')
                f.write(f'Timestamp: {timestamp}\n')
                f.write(f'Total data points: {len(self.all_time_data)}\n')
                f.write(f'Duration: {time_array[-1] - time_array[0]:.2f} seconds\n')
                f.write('='*70 + '\n\n')

                # Overall totals for comparison
                overall_rmse = {'x': 0, 'y': 0, 'z': 0}
                overall_mae = {'x': 0, 'y': 0, 'z': 0}
                overall_total = {'x': 0, 'y': 0, 'z': 0}

                # Overall totals for total distance error
                overall_total_dist_rmse = 0
                overall_total_dist_mae = 0
                overall_total_dist_sum = 0

                for finger in self.finger_names:
                    f.write(f'\n{finger.upper()} FINGER:\n')
                    f.write('-'*40 + '\n')
                    print(f'\n{finger.upper()} FINGER:')

                    for axis in self.axis_names:
                        err = np.array(self.all_error_data[finger][axis])
                        valid = err[~np.isnan(err)]

                        if len(valid) > 0:
                            rmse = np.sqrt(np.mean(valid**2)) * 1000  # Convert to mm
                            mae = np.mean(np.abs(valid)) * 1000  # Convert to mm
                            max_err = np.max(np.abs(valid)) * 1000  # Convert to mm
                            total_err = np.sum(np.abs(valid)) * 1000  # Convert to mm

                            # Accumulate for overall
                            overall_rmse[axis] += rmse
                            overall_mae[axis] += mae
                            overall_total[axis] += total_err

                            line = f'  {axis}: RMSE={rmse:8.3f}mm  MAE={mae:8.3f}mm  Max={max_err:8.3f}mm  Total={total_err:8.1f}mm'
                            f.write(line + '\n')
                            print(line)

                            csv_rows.append([finger, axis, f'{rmse:.3f}', f'{mae:.3f}', f'{max_err:.3f}', f'{total_err:.1f}'])
                        else:
                            f.write(f'  {axis}: No valid data\n')
                            print(f'  {axis}: No valid data')

                    # Calculate total distance error statistics for this finger
                    total_dist_err = np.array(self.all_total_error[finger])
                    valid_total = total_dist_err[~np.isnan(total_dist_err)]
                    if len(valid_total) > 0:
                        t_rmse = np.sqrt(np.mean(valid_total**2)) * 1000
                        t_mae = np.mean(valid_total) * 1000  # Already positive (distance)
                        t_max = np.max(valid_total) * 1000
                        t_sum = np.sum(valid_total) * 1000

                        overall_total_dist_rmse += t_rmse
                        overall_total_dist_mae += t_mae
                        overall_total_dist_sum += t_sum

                        line = f'  TOTAL DIST: RMSE={t_rmse:8.3f}mm  MAE={t_mae:8.3f}mm  Max={t_max:8.3f}mm  Sum={t_sum:8.1f}mm'
                        f.write(line + '\n')
                        print(line)
                        csv_rows.append([finger, 'total_dist', f'{t_rmse:.3f}', f'{t_mae:.3f}', f'{t_max:.3f}', f'{t_sum:.1f}'])

                # Print overall summary
                f.write('\n' + '='*70 + '\n')
                f.write('OVERALL TOTALS (sum across all fingers):\n')
                f.write('-'*40 + '\n')
                print('\n' + '='*70)
                print('OVERALL TOTALS (sum across all fingers):')

                for axis in self.axis_names:
                    line = f'  {axis}: RMSE={overall_rmse[axis]:8.3f}mm  MAE={overall_mae[axis]:8.3f}mm  Total={overall_total[axis]:8.1f}mm'
                    f.write(line + '\n')
                    print(line)

                # Total distance overall
                line = f'  TOTAL DIST: RMSE={overall_total_dist_rmse:8.3f}mm  MAE={overall_total_dist_mae:8.3f}mm  Sum={overall_total_dist_sum:8.1f}mm'
                f.write(line + '\n')
                print(line)

                # Add overall to CSV
                csv_rows.append([])  # Empty row
                csv_rows.append(['OVERALL', 'x', f'{overall_rmse["x"]:.3f}', f'{overall_mae["x"]:.3f}', '', f'{overall_total["x"]:.1f}'])
                csv_rows.append(['OVERALL', 'y', f'{overall_rmse["y"]:.3f}', f'{overall_mae["y"]:.3f}', '', f'{overall_total["y"]:.1f}'])
                csv_rows.append(['OVERALL', 'z', f'{overall_rmse["z"]:.3f}', f'{overall_mae["z"]:.3f}', '', f'{overall_total["z"]:.1f}'])
                csv_rows.append(['OVERALL', 'total_dist', f'{overall_total_dist_rmse:.3f}', f'{overall_total_dist_mae:.3f}', '', f'{overall_total_dist_sum:.1f}'])

                print('='*70 + '\n')
                f.write('='*70 + '\n')

            # Save statistics as CSV for easy comparison
            with open(stats_csv_path, 'w') as f:
                for row in csv_rows:
                    f.write(','.join(str(x) for x in row) + '\n')

            self.get_logger().info(f'Saved statistics: {stats_path}')
            self.get_logger().info(f'Saved statistics CSV: {stats_csv_path}')

        except Exception as e:
            self.get_logger().error(f'Error saving plots: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    plotter = FingertipErrorPlotter()

    # Flag to prevent multiple saves
    save_done = [False]

    def save_and_exit(signum=None, frame=None):
        if save_done[0]:
            # Force exit on second Ctrl+C
            os._exit(0)
        save_done[0] = True

        print('\n\nSaving plots before exit...')
        try:
            plt.ioff()
            plt.close('all')
            print(f'Collected {len(plotter.all_time_data)} data points')
            plotter.save_plots()
            print('Done! Exiting...')
        except Exception as e:
            print(f'Error saving: {e}')
            import traceback
            traceback.print_exc()

        # Force exit
        os._exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        save_and_exit()


if __name__ == '__main__':
    main()
