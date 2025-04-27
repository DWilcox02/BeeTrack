const express = require("express");
const cors = require("cors");
const path = require("path");
const fetch = (url, init) => import("node-fetch").then((module) => module.default(url, init));const app = express();
const port = process.env.PORT || 3000;

const backend_url = "http://127.0.0.1:5001";

// Set static folder
app.use(express.static(path.join(__dirname, "public")));
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Authorization"],
}))
app.use("/data", express.static(path.join(__dirname, "../../data")));
app.use("/output", express.static(path.join(__dirname, "../../output")));

// Set view engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "public/views"));

// Routes
app.get("/", async (req, res) => {
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
  console.log(data);
  res.render("player", data);
});

app.get("/analysis/:filename", async (req, res) => {
  try {
    // Get the first frame from the Flask backend
    const frameResponse = await fetch(`${backend_url}/api/extract_first_frame/${req.params.filename}`);

    if (!frameResponse.ok) {
      throw new Error(`Failed to fetch frame: ${frameResponse.statusText}`);
    }

    const frameData = await frameResponse.json();

    // Generate a unique session ID
    const session_id = Math.random().toString(36).substring(2, 15);

    // Initial points in a rectangle shape (similar to Flask implementation)
    const width = frameData.width;
    const height = frameData.height;

    let points = [
      { x: width * 0.25, y: height * 0.25, color: "red" },
      { x: width * 0.75, y: height * 0.25, color: "green" },
      { x: width * 0.75, y: height * 0.75, color: "blue" },
      { x: width * 0.25, y: height * 0.75, color: "purple" },
    ];

    // TODO: Replace
    points = [
      { x: 1017.8, y: 638.1, color: "red" },
      { x: 1077.8, y: 688.1, color: "green" },
      { x: 1044.4, y: 674.8, color: "blue" },
      { x: 1057.8, y: 659.8, color: "purple" },
    ];

    // Generate Plotly HTML using Plotly.js
    // Note: We'll create a client-side solution instead of server-side rendering
      
    const data = {
      filename: req.params.filename,
      session_id: session_id,
      imageData: frameData.image,
      width: frameData.width,
      height: frameData.height,
      points: JSON.stringify(points),
      plot_html: "", // We'll generate this on client-side
    };

    res.render("frame_analysis", data);
    } catch (error) {
      console.error("Error in analysis route:", error);
      res.status(500).send(`Error processing frame: ${error.message}`);
    }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
