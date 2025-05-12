// Connect to Flask backend
const socket = io("http://127.0.0.1:5001");

// Handle connection
socket.on("connect", () => {
  console.log("Connected to Flask backend");
});

// Handle disconnection
socket.on("disconnect", () => {
  console.log("Disconnected from Flask backend");
});

// Socket API wrapper functions
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

  startNewSession: (session_id, points, video_path, frame_width, frame_height) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "start_new_session",
        {
          session_id: session_id,
          points: points,
          video_path: video_path,
          frame_width: frame_width,
          frame_height: frame_height,
        },
        (response) => {
          if (response.success) {
            resolve(response.session_id);
          } else {
            reject(new Error(response.error || "Failed to start new session"));
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

  // Update all points
  updateAllPoints: (sessionId, points) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "update_all_points",
        {
          session_id: sessionId,
          points: points,
        },
        (response) => {
          if (response.success) {
            resolve(response.points);
          } else {
            reject(new Error(response.error || "Failed to update all points"));
          }
        }
      );
    });
  }
};

// -------------------------------------------------------------------------
// Socket.IO Events
// -------------------------------------------------------------------------

// Listen for job updates
socket.on("job_log", (data) => {
  if (typeof window.handleJobLog === "function") {
    window.handleJobLog(data.job_id, data.message);
  }
});

// Socket event handlers for point updates
socket.on('update_point_response', (result) => {
  console.log("Socket response:", result);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points);
    }

    showStatus("success");
  }
});

socket.on("update_points_with_frame", (result) => {
  console.log("Updating points after interval");
  console.log("New points: " + result.points);
  console.log("New frame index: " + result.frameData.frame_idx);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points, result.frameData);
    }

    showStatus("success");
  }
})

socket.on("validation_request", (data, callback) => {
  console.log("Validation requested by server:", data);

  // Store the request ID
  window.pendingValidationRequestId = data.request_id;

  // Show the validation button
  document.getElementById("validationContinue").style.display = "block";
});


socket.on("update_all_points_response", (result) => {
  console.log("Socket response for all points update:", result);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points);
    }

    showStatus("All points updated successfully!", "success");
  } else {
    showStatus(`Error: ${result.error || "Failed to update points"}`, "error");
  }
});





// Export the API for use in other scripts
window.api = api;
