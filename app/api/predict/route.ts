import { promises as fs } from "node:fs"
import os from "node:os"
import path from "node:path"
import { spawn } from "node:child_process"

import { NextResponse } from "next/server"

export const runtime = "nodejs"

type PythonRunResult = {
  stdout: string
  stderr: string
}

function runProcess(command: string, args: string[], cwd: string): Promise<PythonRunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    })

    let stdout = ""
    let stderr = ""

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString()
    })

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString()
    })

    child.on("error", (err) => {
      reject(err)
    })

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr })
        return
      }

      reject(new Error(`Command failed with exit code ${code}. ${stderr || stdout}`))
    })
  })
}

async function runPythonPrediction(projectRoot: string, imagePath: string): Promise<unknown> {
  const pythonSnippet = [
    "import json, sys, pathlib",
    "root = pathlib.Path(sys.argv[1])",
    "img = pathlib.Path(sys.argv[2])",
    "sys.path.insert(0, str(root / 'scripts'))",
    "from inference import IPCPredictor",
    "result = IPCPredictor().predict_single(str(img), return_details=True)",
    "print(json.dumps(result))",
  ].join("\n")

  const candidates: Array<{ cmd: string; args: string[] }> = [
    { cmd: "python", args: ["-c", pythonSnippet, projectRoot, imagePath] },
    { cmd: "py", args: ["-3", "-c", pythonSnippet, projectRoot, imagePath] },
  ]

  let lastError: unknown = null

  for (const candidate of candidates) {
    try {
      const result = await runProcess(candidate.cmd, candidate.args, projectRoot)
      const trimmed = result.stdout.trim()

      // Python emits progress logs before JSON. Parse from the end line-by-line
      // and return the first valid JSON object found.
      const lines = trimmed
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .reverse()

      for (const line of lines) {
        if (!line.startsWith("{")) {
          continue
        }

        try {
          return JSON.parse(line)
        } catch {
          // Continue scanning previous lines for a valid JSON payload.
        }
      }

      // Fallback: try parsing complete stdout in case payload is multi-line JSON.
      return JSON.parse(trimmed)
    } catch (error) {
      lastError = error
    }
  }

  throw lastError || new Error("Unable to execute Python prediction process.")
}

export async function POST(request: Request) {
  let tempFilePath = ""

  try {
    const formData = await request.formData()
    const image = formData.get("image")

    if (!(image instanceof File)) {
      return NextResponse.json({ error: "No image file uploaded." }, { status: 400 })
    }

    if (!image.type.startsWith("image/")) {
      return NextResponse.json({ error: "Uploaded file must be an image." }, { status: 400 })
    }

    if (image.size > 10 * 1024 * 1024) {
      return NextResponse.json({ error: "Image is too large. Max size is 10 MB." }, { status: 400 })
    }

    const extension = image.name.split(".").pop() || "png"
    tempFilePath = path.join(os.tmpdir(), `fir-upload-${Date.now()}.${extension}`)
    const buffer = Buffer.from(await image.arrayBuffer())

    await fs.writeFile(tempFilePath, buffer)

    const prediction = await runPythonPrediction(process.cwd(), tempFilePath)
    return NextResponse.json(prediction)
  } catch (error) {
    const message = error instanceof Error ? error.message : "Prediction failed due to an unknown error."
    return NextResponse.json(
      {
        error: "Unable to run prediction. Ensure Python dependencies are installed and model files exist.",
        details: message,
      },
      { status: 500 }
    )
  } finally {
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath)
      } catch {
        // File cleanup best effort
      }
    }
  }
}
