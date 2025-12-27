
import cv2
import mediapipe as mp
import numpy as np

class FeatureExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the MediaPipe Hands module.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, image):
        """
        Processes an image/frame and extracts hand landmarks.
        Returns:
            np.array: A flat array of 42 float values (21 landmarks * 2 coords).
                      Order: [x0, y0, ... x20, y20].
                      Returns zero array if no hand is detected.
        """
        if image is None:
            return np.zeros(42)

        # Flip the image horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            # For IPNHand migration, we only use one hand. 
            # We take the first detected hand.
            hand_landmarks = results.multi_hand_landmarks[0]
            
            lm_list = []
            for lm in hand_landmarks.landmark:
                # MediaPipe returns normalized coordinates [0.0, 1.0]
                lm_list.extend([lm.x, lm.y])
            
            return np.array(lm_list)
                
        else:
            # Return zeros if no hand found to keep time alignment
            return np.zeros(42)

    def draw_landmarks(self, image, results):
        """
        Helper to draw landmarks on the image for debugging/UI.
        Note: requires 'results' object from self.hands.process, 
        which we don't return in extract_landmarks for speed. 
        Use this only when visualization is needed.
        """
        # This implementation serves as a placeholder or needs the raw results 
        # to be passed back if visualizations are required.
        pass

    def close(self):
        self.hands.close()
