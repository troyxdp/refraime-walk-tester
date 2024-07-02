import tkinter as tk
from tkinter import ttk
import yaml
import os
# import sys

from walk_tester import WalkTester

class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Create container
        container = tk.Frame(self)  
        container.pack(side="top", fill="both", expand=True, padx=15, pady=15) 
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # Initialize frames to an empty array
        self.frames = {}

        # Add Frames to frame list
        for F in (AddURLsPage,):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        # Display first frame
        self.show_frame(AddURLsPage)

    # Method for navigating between frames
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class AddURLsPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        with open('config/streams.yaml', 'r') as f:
            self.cameras = yaml.load(f, Loader=yaml.SafeLoader)

        self.curr_cam_num = -1

        self.models = os.listdir('weights')
        for i in range(len(self.models)):
            self.models[i] = os.path.splitext(self.models[i])[0]

        # PAGE TITLE SECTION
        # Page title label
        page_title_label = tk.Label(
            self,
            text="Walk Tester",
            font=('Times', 24, 'bold')
        )
        page_title_label.grid(row=1, column=2)

        # ADD CAMERAS SECTION
        # Label for section to add cameras
        add_cameras_label = tk.Label(
            self,
            text="Add Cameras:",
            font=('Times', 18, 'bold')
        )
        add_cameras_label.grid(row=2, column=1, sticky=tk.W)

        # Variable to store camera name
        self.camera_name = tk.StringVar()
        # Label for camera name text field
        camera_name_box_label = tk.Label(
            self,
            text="Camera Name: ",
            font=('Times', 14)
        )
        camera_name_box_label.grid(row=3, column=1, sticky=tk.E)
        # Text field for camera name entry
        self.camera_name_entry = tk.Entry(
            self,
            textvariable=self.camera_name
        )
        self.camera_name_entry.grid(row=3, column=2, sticky=tk.W)

        # Variable to store RTSP URL
        self.rtsp_url = tk.StringVar()
        # Label for RTSP URL text field
        rtsp_url_box_label = tk.Label(
            self,
            text="RTSP URL: ",
            font=('Times', 14)
        )
        rtsp_url_box_label.grid(row=4, column=1, sticky=tk.E)
        # Text field for camera RTSP URL entry
        self.rtsp_url_entry = tk.Entry(
            self,
            textvariable=self.rtsp_url
        )
        self.rtsp_url_entry.grid(row=4, column=2, sticky=tk.W)

        # Button to add RTSP URL
        self.add_camera_button = tk.Button(
            self,
            text="Add",
            command=self.add_camera,
            padx=10,
            pady=5
        )
        self.add_camera_button.grid(row=5, column=2, sticky=tk.W)

        # EDIT CAMERA SECTION
        # Label for this section of the page
        edit_cameras_label = tk.Label(
            self,
            text="Edit Camera:",
            font=('Times', 18, 'bold')
        )
        edit_cameras_label.grid(row=6, column=1, sticky=tk.W)

        # Label for combobox
        camera_combobox_label = tk.Label(
            self,
            text="Camera: ",
            font=('Times', 14)
        )
        camera_combobox_label.grid(row=7, column=1, sticky=tk.E)

        # Combobox for selecting which camera to edit
        self.camera_selected = tk.StringVar()
        self.camera_combobox = ttk.Combobox(
            self,
            values=list(self.cameras.keys()),
            state='readonly',
            textvariable=self.camera_selected
        )
        self.camera_combobox.grid(row=7, column=2, sticky=tk.W)
        self.camera_combobox.bind(
            '<<ComboboxSelected>>',
            self.on_camera_selected_change
        )

        # Label for camera name editting text field
        edit_camera_name_label = tk.Label(
            self,
            text="Camera Name: ",
            font=('Times', 14)
        )
        edit_camera_name_label.grid(row=8, column=1, sticky=tk.E)
        # Text field for editting camera name and string var for storing it
        self.new_camera_name = tk.StringVar(value='')
        self.edit_camera_name_entry = tk.Entry(
            self,
            textvariable=self.new_camera_name
        )
        self.edit_camera_name_entry.grid(row=8, column=2, sticky=tk.W)

        # Label for RTSP URL editting text field
        edit_rtsp_url_label = tk.Label(
            self,
            text="RTSP URL: ",
            font=('Times', 14)
        )
        edit_rtsp_url_label.grid(row=9, column=1, sticky=tk.E)
        # Text field for editting camera RTSP URL and string var for storing it
        self.new_rtsp_url = tk.StringVar(value='')
        self.edit_rtsp_url_entry = tk.Entry(
            self,
            textvariable=self.new_rtsp_url
        )
        self.edit_rtsp_url_entry.grid(row=9, column=2, sticky=tk.W)

        # Set the combobox and subsequent fields to some value
        if len(self.cameras.keys()) > 0:
            self.camera_combobox.current(0)
            initial_cam = self.camera_combobox.get()
            for cam in self.camera_combobox['values']:
                if cam == initial_cam:
                    self.camera_selected.set(cam)
                    self.new_camera_name.set(cam)
                    self.new_rtsp_url.set(self.cameras[cam])
                    break

        # Buttons for editting camera details
        self.apply_edit_button = tk.Button(
            self,
            text="Edit",
            command=self.edit_camera,
            padx=10,
            pady=5
        )
        self.apply_edit_button.grid(row=10, column=2, sticky=tk.W)
        # Buttons for deleting camera
        self.delete_camera_button = tk.Button(
            self,
            text="Delete",
            command=self.delete_camera,
            padx=10,
            pady=5
        )
        self.delete_camera_button.grid(row=11, column=2, sticky=tk.W)

        # MODEL SELECTOR
        # Label for section
        model_selector_label = tk.Label(
            self,
            text="Model Selector:",
            font=('Times', 18, 'bold')
        )
        model_selector_label.grid(row=12, column=1, sticky=tk.W)
        # Label for combobox
        model_combobox_label = tk.Label(
            self,
            text="Model: ",
            font=('Times', 14)
        )
        model_combobox_label.grid(row=13, column=1, sticky=tk.E)
        # Combobox and string var to store value in it
        self.model_selected = tk.StringVar(value='')
        model_combobox = ttk.Combobox(
            self,
            values=list(self.models),
            state='readonly',
            textvariable=self.model_selected
        )
        model_combobox.grid(row=13, column=2, sticky=tk.W)
        model_combobox.current(0)

        # Button to save config
        save_config_button = tk.Button(
            self,
            text="Save",
            command=self.save_config,
            padx=10,
            pady=5
        )
        save_config_button.grid(row=15, column=1, sticky=tk.W, pady=5)

        # Button to start walk test/resume on next camera and display info about what cam is being tested
        self.start_test_button = tk.Button(
            self,
            text="Start",
            command=self.start_test,
            padx=10,
            pady=5
        )
        self.test_info = tk.StringVar()
        self.test_info_label = tk.Label(
            self,
            textvariable=self.test_info,
            font=('Times', 12)
        )
        # self.start_test_button.grid(row=13, column=1, sticky=tk.W)
        self.stop_test_button = tk.Button(
            self,
            text="Stop Testing",
            command=self.stop_test,
            padx=5,
            pady=5
        )

        self.walk_tester = None



    def add_camera(self):
        # Get name and rtsp url of camera to add
        camera_name = self.camera_name.get().strip()
        rtsp_url = self.rtsp_url.get().strip()

        if not (camera_name == '' or rtsp_url == ''):
            # Add camera
            self.cameras[f'{camera_name}'] = rtsp_url
            self.camera_combobox['values'] = list(self.cameras.keys())
            self.camera_combobox.current(0)

            # Clear the entry fields
            self.camera_name.set('')
            self.rtsp_url.set('')

            # Choose something on the camera select combobox
            for i, cam in enumerate(self.camera_combobox['values']):
                if cam == camera_name:
                    self.camera_combobox.current(i)
                    self.camera_selected.set(cam)
                    self.new_camera_name.set(camera_name)
                    self.new_rtsp_url.set(rtsp_url)
                    break
        else:
            print("Error: please enter a valid camera name and RTSP URL")

    def edit_camera(self):
        if len(self.cameras.keys()) > 0:
            # Remove old camera details
            camera_to_edit = self.camera_selected.get()
            
            # Get updated info
            camera_name = self.new_camera_name.get().strip()
            rtsp_url = self.new_rtsp_url.get().strip()

            if not (camera_name == '' or rtsp_url == ''):
                # Remove old camera details
                self.cameras.pop(camera_to_edit)

                # Add it to cameras dict and values of combobox
                self.cameras[f'{camera_name}'] = rtsp_url
                self.camera_combobox['values'] = list(self.cameras.keys())

                # Print for debugging purposes
                print(self.cameras)

                # Change data that is in the fields
                for i, cam in enumerate(self.camera_combobox['values']):
                    if cam == camera_name:
                        self.new_camera_name.set(cam)
                        self.new_rtsp_url.set(self.cameras[cam])
                        self.camera_combobox.current(i)
                        break
            else:
                print("Error: please enter a valid camera name and RTSP URL")
        else:
            print("Error: no cameras to edit")

    def delete_camera(self):
        camera_to_delete = self.camera_selected.get()
        if not camera_to_delete.strip() == '':
            self.cameras.pop(camera_to_delete)
            self.camera_combobox['values'] = list(self.cameras.keys())
            if len(self.cameras.keys()) > 0:
                self.camera_combobox.current(0)
                self.new_camera_name.set(self.camera_selected.get())
                self.new_rtsp_url.set(self.cameras[f'{self.camera_selected.get()}'])
            else:
                self.new_camera_name.set('')
                self.new_rtsp_url.set('')
                self.camera_selected.set('')
            print(self.cameras)
        else:
            print("Error: please select a camera to delete")

    def on_camera_selected_change(self, event):
        camera_name = self.camera_selected.get()
        rtsp_url = self.cameras[f'{camera_name}']
        self.new_camera_name.set(camera_name)
        self.new_rtsp_url.set(rtsp_url)

    def save_config(self):
        if len(self.cameras.keys()) > 0:
            # Save config to yaml file to ensure persistence
            print("Saving config...")
            with open('config/streams.yaml', 'w') as f:
                yaml.dump(self.cameras, f)

            # Initialize walk tester object
            if self.model_selected.get().strip() == '':
                self.walk_tester = WalkTester()
            else:
                model_name = f'weights/{self.model_selected.get().strip()}.pt'
                self.walk_tester = WalkTester(model_name=model_name)

            # Only display start button now
            self.start_test_button.grid(row=16, column=1, sticky=tk.W)
            self.start_test_button.configure(text="Start")

            # Set text fields to disabled to prevent modification of data
            self.camera_name_entry.config(state='readonly')
            self.rtsp_url_entry.config(state='readonly')
            self.edit_camera_name_entry.config(state='readonly')
            self.edit_rtsp_url_entry.config(state='readonly')

            # Set add, delete, and edit buttons to disabled to prevent modification of data
            self.add_camera_button.config(state='disabled')
            self.apply_edit_button.config(state='disabled')
            self.delete_camera_button.config(state='disabled')
        else:
            print("Error: no cameras added for testing")

    def start_test(self):
        if self.curr_cam_num < len(self.cameras.keys()) - 1:
            self.start_test_button.configure(text="Continue")
            # Hide stop button
            self.stop_test_button.grid_forget()

            # Get RTSP URL and cam name to run test on
            self.curr_cam_num += 1
            curr_cam = list(self.cameras.keys())[self.curr_cam_num].strip()
            curr_rtsp = list(self.cameras.values())[self.curr_cam_num].strip()

            # Display info about test
            test_info_text = f"Running test for '{curr_cam}' camera"
            self.test_info.set(test_info_text)
            self.test_info_label.grid(row=16, column=2, sticky=tk.W)
            print(test_info_text)

            # Run test - stops when 'q' is pressed
            self.walk_tester.run_processor(curr_rtsp, curr_cam)

            # For debugging
            print("Finished test for camera {}\n".format(curr_cam))

            # Unhide the stop test button
            self.stop_test_button.grid(row=17, column=1, sticky=tk.W)
            self.test_info_label.grid_forget()

            # If it is just done the final camera, set the button text to "Finish"
            if self.curr_cam_num == len(self.cameras.keys()) - 1:
                self.start_test_button.configure(text="Finish")
        else:
            print("Done testing")

            # Remove buttons used during testing
            self.start_test_button.grid_forget()
            self.stop_test_button.grid_forget()

            # Set text fields to normal to allow modification of data
            self.camera_name_entry.config(state='normal')
            self.rtsp_url_entry.config(state='normal')
            self.edit_camera_name_entry.config(state='normal')
            self.edit_rtsp_url_entry.config(state='normal')

            # Set add, delete, and edit buttons to normal to allow modification of data
            self.add_camera_button.config(state='normal')
            self.apply_edit_button.config(state='normal')
            self.delete_camera_button.config(state='normal')

            # Reset curr_cam_num so tests can be run again
            self.curr_cam_num = -1

    def stop_test(self):
        # Hide all the buttons and reset curr_cam_num
        self.curr_cam_num = -1
        self.start_test_button.grid_forget()
        self.stop_test_button.grid_forget()

        # Set text fields to normal to allow modification of data
        self.camera_name_entry.config(state='normal')
        self.rtsp_url_entry.config(state='normal')
        self.edit_camera_name_entry.config(state='normal')
        self.edit_rtsp_url_entry.config(state='normal')

        # Set add, delete, and edit buttons to normal to allow modification of data
        self.add_camera_button.config(state='normal')
        self.apply_edit_button.config(state='normal')
        self.delete_camera_button.config(state='normal')



if __name__ == '__main__':
    app = tkinterApp()
    app.mainloop()