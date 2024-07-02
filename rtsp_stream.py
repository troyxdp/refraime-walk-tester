import logging
import sys
import multiprocessing as mp
from enum import Enum
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import ffmpeg

from signal import signal
from signal import SIGTERM

Gst.init(None)

class StreamMode(Enum):
    INIT_STREAM = 1
    SETUP_STREAM = 1
    READ_STREAM = 2


class StreamCommands(Enum):
    FRAME = 1
    ERROR = 2
    HEARTBEAT = 3
    RESOLUTION = 4
    STOP = 5


class StreamCapture(mp.Process):

    def __init__(self, camera, stop, outQueue, framerate):
        """
        Initialize the stream capturing process
        link - rstp link of stream
        stop - to send commands to this process
        outPipe - this process can send commands outside
        """

        super().__init__()
        self.streamLink = camera['rtsp_url']
        self.camera_id = camera['camera_id']
        self.stop = stop
        self.outQueue = outQueue
        self.framerate = framerate
        self.currentState = StreamMode.INIT_STREAM
        self.pipeline = None
        self.source = None
        self.decode = None
        self.convert = None
        self.sink = None
        self.image_arr = None
        self.newImage = False
        self.frame1 = None
        self.frame2 = None
        self.num_unexpected_tot = 40
        self.unexpected_cnt = 0

    # handle signal
    def handler(self, sig, frame):
        self.stop.set()
        self.pipeline.set_state(Gst.State.NULL)      
        
        logging.info("Termintating child processes from rtsp_stream .....")  

        sys.exit(0)

    def replace_special_characters(self, uri):
        # Splitting the URI into components
        parts = uri.split('@')

        # Checking if the URI has a password
        if ':' in parts[0]:
            # Splitting the user info part to separate out the password
            protocol, user_info = parts[0].split('://')

            # Replacing special characters in the password
            username, password = user_info.split(':', 1)
            replaced_password = password.replace('#', '%23').replace('@', '%40')

            # Reconstructing the URI with the replaced password
            #modified_uri = f"{parts[0]}:{userinfo.split(':')[0]}:{replaced_password}@{hostport}:{':'.join(parts[2:])}"
            modified_uri = f"{protocol}://{username}:{replaced_password}@{parts[1]}"
            return modified_uri
        else:
            return uri

    def probe_cam(self, rtsp_url):

        try:
            rtsp_url = self.replace_special_characters(rtsp_url)

            args = {"rtsp_transport": "tcp"}

            
            probe = ffmpeg.probe(rtsp_url, **args)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            #fps = int(video_info['r_frame_rate'].split('/')[0])
            video_codec = video_info['codec_name']

            return video_codec 
        
        except ffmpeg.Error as e:
            logging.error(f"Error connecting to {rtsp_url}: {e.stderr.decode()}")
            return ""

    def gst_to_opencv(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Print Height, Width and Format
        # print(caps.get_structure(0).get_value('format'))
        # print(caps.get_structure(0).get_value('height'))
        # print(caps.get_structure(0).get_value('width'))

        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
             caps.get_structure(0).get_value('width'),
             3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        arr = self.gst_to_opencv(sample)
        self.image_arr = arr
        self.newImage = True
        return Gst.FlowReturn.OK

    def run(self):
        
        signal(SIGTERM, self.handler)

        try:
            video_codec = self.probe_cam(self.streamLink)

            # Create the empty pipeline
            # video/x-raw,width=[1,640],height=[1,480],pixel-aspect-ratio=1/1

            if video_codec == "h264":
                # self.pipeline = Gst.parse_launch(
                #     'rtspsrc name=m_rtspsrc ! rtph264depay name=m_rtphdepay ! avdec_h264 name=m_avdech ! videoconvertscale name=m_videoconvertscale ! video/x-raw,width=640,height=480 ! videorate name=m_videorate ! appsink name=m_appsink'
                # )
                self.pipeline = Gst.parse_launch(
                    'rtspsrc name=m_rtspsrc ! rtph264depay name=m_rtphdepay ! avdec_h264 name=m_avdech ! videoconvert name=m_videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! videorate name=m_videorate ! appsink name=m_appsink'
                )

            else:
                # self.pipeline = Gst.parse_launch(
                #     'rtspsrc name=m_rtspsrc ! rtph265depay name=m_rtphdepay ! avdec_h265 name=m_avdech ! videoconvertscale name=m_videoconvertscale ! video/x-raw,width=640,height=480 ! videorate name=m_videorate ! appsink name=m_appsink'
                # )
                self.pipeline = Gst.parse_launch(
                    'rtspsrc name=m_rtspsrc ! rtph265depay name=m_rtphdepay ! avdec_h265 name=m_avdech ! videoconvert name=m_videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! videorate name=m_videorate ! appsink name=m_appsink'
                )

            # source params
            self.source = self.pipeline.get_by_name('m_rtspsrc')
            self.source.set_property('latency', 1000)
            self.source.set_property('location', self.streamLink)

            self.source.set_property('protocols', 'tcp')
            self.source.set_property('retry', 50)
            self.source.set_property('timeout', 1000000)
            self.source.set_property('tcp-timeout', 5000000)
            self.source.set_property('drop-on-latency', 'true')

            # decode params
            self.decode = self.pipeline.get_by_name('m_avdech')
            self.decode.set_property('max-threads', 1)
            self.decode.set_property('output-corrupt', 'false')

            # convert params
            self.convert = self.pipeline.get_by_name('m_videoconvertscale')


            #framerate parameters
            self.framerate_ctr = self.pipeline.get_by_name('m_videorate')
            self.framerate_ctr.set_property('max-rate', self.framerate/1)
            self.framerate_ctr.set_property('drop-only', 'true')

            # sink params
            self.sink = self.pipeline.get_by_name('m_appsink')

            # Maximum number of nanoseconds that a buffer can be late before it is dropped (-1 unlimited)
            # flags: readable, writable
            # Integer64. Range: -1 - 9223372036854775807 Default: -1
            self.sink.set_property('max-lateness', 500000000)

            # The maximum number of buffers to queue internally (0 = unlimited)
            # flags: readable, writable
            # Unsigned Integer. Range: 0 - 4294967295 Default: 0
            self.sink.set_property('max-buffers', 5)

            # Drop old buffers when the buffer queue is filled
            # flags: readable, writable
            # Boolean. Default: false
            self.sink.set_property('drop', 'true')

            # Emit new-preroll and new-sample signals
            # flags: readable, writable
            # Boolean. Default: false
            self.sink.set_property('emit-signals', True)

            # The allowed caps for the sink pad
            # flags: readable, writable
            # Caps (NULL)
            
            caps = Gst.caps_from_string(
                'video/x-raw, format=(string){RGB}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}')
            self.sink.set_property('caps', caps)


            if not self.source or not self.sink or not self.pipeline:
                logging.error(f"Not all elements could be created for camera id {self.camera_id}")
                self.stop.set()

            self.sink.connect("new-sample", self.new_buffer, self.sink)

            # Start playing
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logging.error(f"Unable to set the pipeline to the playing state for camera id {self.camera_id}")
                self.stop.set()

            # Wait until error or EOS
            bus = self.pipeline.get_bus()

            while True:

                if self.stop.is_set():
                    # logging.error(f"Stopping CAM Stream by main process for camera id {self.camera_id}")
                    break

                message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
                if self.image_arr is not None and self.newImage is True:

                    if not self.outQueue.full():

                        # print("\r adding to queue of size{}".format(self.outQueue.qsize()), end='\r')
                        self.outQueue.put((StreamCommands.FRAME, self.image_arr), block=False)

                    self.image_arr = None
                    self.unexpected_cnt = 0


                if message:
                    if message.type == Gst.MessageType.ERROR:
                        err, debug = message.parse_error()
                        logging.error(f"Error received from element {message.src.get_name()}: {err}") 
                        logging.error(f"Debugging information for camera id {self.camera_id}: {debug}"  )
                        break
                    elif message.type == Gst.MessageType.EOS:
                        logging.warning(f"End-Of-Stream reached for camera id {self.camera_id}")
                        break
                    elif message.type == Gst.MessageType.STATE_CHANGED:
                        if isinstance(message.src, Gst.Pipeline):
                            old_state, new_state, pending_state = message.parse_state_changed()
                            logging.info(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick} for camera id {self.camera_id}") 
                    else:
                        logging.debug(f"Unexpected message received for camera id {self.camera_id}")
                        self.unexpected_cnt = self.unexpected_cnt + 1
                        if self.unexpected_cnt == self.num_unexpected_tot:
                            break


            # logging.error(f"Terminating camera pipeline for camera id {self.camera_id}")
            self.stop.set()
            self.pipeline.set_state(Gst.State.NULL)
            print("Emptying and closing queue...")
            while not self.outQueue.empty():
                try:
                    print("Emptying...")
                    self.outQueue.get(block=False)
                except self.outQueue.Empty:
                    break
            self.outQueue.cancel_join_thread()
            self.outQueue.close()
            print(f"Terminating camera pipeline for camera id {self.camera_id}")
            sys.exit(0)

        except Exception as error:
            self.stop.set()
            logging.error("Error occured in starting pipeline for stream{}: {}".format(self.camera_id, error))
            sys.exit(1)

