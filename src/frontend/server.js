const express = require("express");
const path = require("path");
const app = express();
const port = process.env.PORT || 3000;

// Set static folder
app.use(express.static(path.join(__dirname, "public")));

// Set view engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "public/views"));

// Routes
app.get("/", (req, res) => {
  // Mock data - in production this would come from the Flask backend
  const data = {
    videos: ["video1.mp4", "video2.mp4", "folder/video3.mp4"],
    processed_videos: ["processed_video1.mp4", "processed_video2.mp4"],
  };
  res.render("index", data);
});

app.get("/video/:filename", (req, res) => {
  // In production, data would be fetched from Flask
  const data = {
    filename: req.params.filename,
    processed_filename: null,
    point_cloud_available: true,
  };
  res.render("player", data);
});

app.get("/analysis/:filename", (req, res) => {
  // In production, data would be fetched from Flask
  const data = {
    filename: req.params.filename,
    session_id: Math.random().toString(36).substring(2, 15),
  };
  res.render("frame_analysis", data);
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
