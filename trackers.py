import cv2
import csv

def initialize_trackers(frame, detector):
    """
    Initialize trackers based on the detected keypoints
    """
    trackers = []

    # Detect blobs in the original frame
    keypoints = detector.detect(frame)

    print(f"Number of keypoints detected: {len(keypoints)}")

    for keypoint in keypoints:
        # ================ Initialize a tracker for each blob (choose one) ================
        # tracker = cv2.legacy.TrackerMOSSE_create()
        # tracker = cv2.legacy.TrackerBoosting_create()
        # tracker = cv2.legacy.TrackerMIL_create()
        # tracker = cv2.legacy.TrackerKCF_create()
        # tracker = cv2.legacy.TrackerTLD_create()
        # tracker = cv2.legacy.TrackerMedianFlow_create()
        tracker = cv2.legacy.TrackerCSRT_create()
        # =================================================================================

        # ================ Bounding box for the tracker (choose one) ======================
        # bbox = (keypoint.pt[0]-20, keypoint.pt[1]-20, 40, 40) # 40 x 40
        bbox = (keypoint.pt[0]-30, keypoint.pt[1]-30, 60, 60) # 60 x 60
        # bbox = (keypoint.pt[0]-40, keypoint.pt[1]-40, 80, 80) # 80 x 80
        # =================================================================================

        tracker.init(frame, bbox)
        trackers.append(tracker)
    return trackers


def track_and_write_points(cap, trackers, writer):
    """
    Track points in the video and write their positions to the CSV
    """
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                print(f"Tracker update success for frame {frame_counter}")
                # Write frame number, x and y to CSV
                writer.writerow([frame_counter, bbox[0], bbox[1]])
            else:
                print(f"Tracker update failed for frame {frame_counter}")

def main():
    # ================ Video to load (choose one) =========================================
    # cap = cv2.VideoCapture('bml-walker.mp4')
    cap = cv2.VideoCapture('bml-walker-side.mp4')
    # =====================================================================================

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.filterByColor = True
    params.blobColor = 255


    # Create BlobDetector
    detector = cv2.SimpleBlobDetector_create(params)

    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    trackers = initialize_trackers(frame, detector)

    # Open CSV file and initialize writer
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        track_and_write_points(cap, trackers, writer)

    cap.release()

if __name__ == "__main__":
    main()
