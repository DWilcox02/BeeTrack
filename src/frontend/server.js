const express = require("express");
const path = require("path");
const app = express();
const port = process.env.PORT || 3000;

const backend_url = "http://127.0.0.1:5001"

// Set static folder
app.use(express.static(path.join(__dirname, "public")));
app.use("/data", express.static(path.join(__dirname, "../../data")));
app.use("/output", express.static(path.join(__dirname, "../../output")));


// Set view engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "public/views"));

// Routes
app.get("/", async (req, res) => {
  // Mock data - in production this would come from the Flask backend
  try {
    const videos = 
      await fetch(backend_url + "/api/videos")
            .then((response) => response.json())
    const processed_videos = 
      await fetch(backend_url + "/api/processed_videos")
      .then((response) => response.json());
    const data = {
      videos: videos,
      processed_videos: processed_videos
    }
    res.render("index", data);
  } catch (error) {
    console.error("Error fetching videos from backend: ", error);
    res.status(500).send("Error fetching videos from backend");
  }
});

app.get("/video/:filename", (req, res) => {
  const data = {
    filename: req.params.filename,
    processed_filename: null,
    point_cloud_available: true,
  };
  console.log(data)
  res.render("player", data);
});

app.get("/analysis/:filename", (req, res) => {
  const data = {
    filename: req.params.filename,
    session_id: Math.random().toString(36).substring(2, 15),
  };
  res.render("frame_analysis", data);
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
