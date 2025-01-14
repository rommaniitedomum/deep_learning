const express = require("express");
const cors = require("cors");
const path = require("path");
const spawn = require("child_process").spawn;
const app = express();

const PORT = 8000;

app.use(cors());
app.use(express.json());

// 루트 경로 호출
app.get("/", (request, response) => {
  console.log(`Server is running on port ${PORT}`);
});

app.post("/abalone", (request, response) => {
  try {
    const scriptPath = path.join(__dirname, "app.py");
    const inputData = request.body;
    const result = spawn(pythonExePath, [
      scriptPath,
      JSON.stringify(inputData),
    ]);

    let responseData = "";

    result.stdout.on("data", (data) => {
      responseData = data
        .toString()
        .split("\n")
        .filter((line) => line.trim())
        .pop();
    });

    result.stderr.on("data", (data) => {
      console.error(`Python 스크립트 에러: ${data}`);
    });

    result.on("close", (code) => {
      if (code === 0) {
        try {
          responseData = responseData.trim();
          const jsonData = JSON.parse(responseData);
          response.status(200).json(jsonData);
        } catch (parseError) {
          console.error(
            "JSON 파싱 오류:",
            parseError,
            "Raw data:",
            responseData
          );
          response.status(500).json({
            error: "JSON 파싱 오류",
            rawData: responseData,
          });
        }
      } else {
        response.status(500).json({ error: "Python script failed" });
      }
    });
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "Internal server error" });
  }
});

const pythonExePath = path.join("python");

app.get("/get_text", (request, response) => {
  const scriptPath = path.join(__dirname, "app.py");
  const result = spawn(pythonExePath, [scriptPath]);

  let = resData = "";

  result.stdout.on("data", (data) => {
    resData += data.toString();
  });

  result.on("close", (code) => {
    if (code === 0) {
      // const jsonData = JSON.parse(resData);
      response.json(resData);
    } else {
      response.status(500).json({ error: "Python script failed" });
    }
  });
});

// 서버 실행
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
