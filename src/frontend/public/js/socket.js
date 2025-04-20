// Connect to Flask backend
const socket = io("http://localhost:5000");

// Handle connection
socket.on("connect", () => {
  console.log("Connected to Flask backend");
});

// Handle disconnection
socket.on("disconnect", () => {
  console.log("Disconnected from Flask backend");
});

// API wrapper functions
const api = {
  // Get all videos
  getVideos: () => {
    return new Promise((resolve, reject) => {
      socket.emit("get_videos", {}, (response) => {
        if (response.success) {
          resolve(response.data);
        } else {
          reject(new Error(response.error || "Failed to get videos"));
        }
      });
    });
  },

  // Process video
  processVideo: (videoPath) => {
    return new Promise((resolve, reject) => {
      socket.emit("process_video", { video_path: videoPath }, (response) => {
        if (response.success) {
          resolve(response.job_id);
        } else {
          reject(new Error(response.error || "Failed to start processing"));
        }
      });
    });
  },

  // Process video with points
  processVideoWithPoints: (videoPath, sessionId) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "process_video_with_points",
        {
          video_path: videoPath,
          session_id: sessionId,
        },
        (response) => {
          if (response.success) {
            resolve(response.job_id);
          } else {
            reject(new Error(response.error || "Failed to start processing with points"));
          }
        }
      );
    });
  },

  // Update point
  updatePoint: (sessionId, pointIndex, x, y) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "update_point",
        {
          session_id: sessionId,
          point_index: pointIndex,
          x: x,
          y: y,
        },
        (response) => {
          if (response.success) {
            resolve(response.plot_data);
          } else {
            reject(new Error(response.error || "Failed to update point"));
          }
        }
      );
    });
  },

  // Save points
  savePoints: (sessionId) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "save_points",
        {
          session_id: sessionId,
        },
        (response) => {
          if (response.success) {
            resolve(response.points);
          } else {
            reject(new Error(response.error || "Failed to save points"));
          }
        }
      );
    });
  },

  // Check job status
  checkJobStatus: (jobId) => {
    return new Promise((resolve, reject) => {
      socket.emit("job_status", { job_id: jobId }, (response) => {
        if (response.success) {
          resolve(response.result);
        } else {
          reject(new Error(response.error || "Failed to get job status"));
        }
      });
    });
  },
};

// Listen for job updates
socket.on("job_log", (data) => {
  if (typeof window.handleJobLog === "function") {
    window.handleJobLog(data.job_id, data.message);
  }
});

// Export the API for use in other scripts
window.api = api;
