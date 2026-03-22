"use client"

import { ChangeEvent, FormEvent, useMemo, useState } from "react"

type TopPrediction = {
  rank: number
  ipc_code: string
  description: string
  probability: number
  percentage: string
}

type ImportantFeature = {
  rank: number
  feature_name: string
  importance: number
  importance_percentage: string
}

type PredictionResult = {
  predicted_ipc: string
  ipc_description: string
  confidence: number
  confidence_percentage: string
  refined_image_base64?: string
  refined_image_mime?: string
  best_refined_variant?: "original" | "readability" | "stain_reduced" | "deblurred" | "deskewed"
  prediction_status?: "final_verdict"
  trust_level?: "refined_pipeline"
  is_reliable?: boolean
  user_recommendation?: string
  selected_image_variant?: "original" | "readability" | "stain_reduced" | "deblurred" | "deskewed"
  image_quality?: {
    used_variant?: "original" | "readability" | "stain_reduced" | "deblurred" | "deskewed"
    original?: {
      quality_score?: number
    }
    readability?: {
      quality_score?: number
    }
    stain_reduced?: {
      quality_score?: number
    }
    deblurred?: {
      quality_score?: number
    }
    deskewed?: {
      quality_score?: number
    }
    best_refined?: {
      quality_score?: number
    }
  }
  certainty_diagnostics?: {
    quality_score?: number
    confidence_gap?: number
    reasons?: string[]
  }
  top_k_predictions?: TopPrediction[]
  important_features?: ImportantFeature[]
  error?: string
  status?: string
}

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const canSubmit = useMemo(() => Boolean(selectedFile) && !isLoading, [selectedFile, isLoading])

  function getCertaintyLabel(confidence: number) {
    if (confidence >= 0.8) {
      return "Very high certainty"
    }
    if (confidence >= 0.6) {
      return "High certainty"
    }
    if (confidence >= 0.4) {
      return "Medium certainty"
    }
    return "Low certainty"
  }

  function getTrustPill(resultData: PredictionResult) {
    return {
      label: "Final verdict from refined image pipeline",
      className: "border-emerald-300/40 bg-emerald-500/20 text-emerald-100",
    }
  }

  function getPlainCategory(description: string) {
    return description
      .replace(/^Punishment for\s+/i, "")
      .replace(/^Attempt to\s+/i, "attempted ")
      .replace(/^Act\s+/i, "")
      .replace(/^Causing\s+/i, "causing ")
      .trim()
  }

  function getUserGuidance(resultData: PredictionResult) {
    const confidence = resultData.confidence ?? 0
    const secondBest = resultData.top_k_predictions?.[1]?.probability ?? 0
    const confidenceGap = confidence - secondBest
    const diagnostics = resultData.certainty_diagnostics
    const reasons = diagnostics?.reasons ?? []
    const qualityScore =
      diagnostics?.quality_score ??
      resultData.image_quality?.best_refined?.quality_score ??
      resultData.image_quality?.deskewed?.quality_score ??
      resultData.image_quality?.deblurred?.quality_score ??
      resultData.image_quality?.stain_reduced?.quality_score ??
      resultData.image_quality?.readability?.quality_score ??
      0
    const variantUsed = resultData.image_quality?.used_variant ?? resultData.selected_image_variant ?? "original"

    if (reasons.length > 0) {
      return `Refinement and fusion completed. Final scoring used the ${variantUsed} variant most strongly, with quality score ${qualityScore.toFixed(2)}.`
    }

    return `Refinement and fusion completed. Final scoring used the ${variantUsed} variant most strongly, with quality score ${qualityScore.toFixed(2)}.`
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setResult(null)
    setErrorMessage(null)

    if (!file) {
      setSelectedFile(null)
      setPreviewUrl(null)
      return
    }

    if (!file.type.startsWith("image/")) {
      setSelectedFile(null)
      setPreviewUrl(null)
      setErrorMessage("Please choose a valid image file.")
      return
    }

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!selectedFile) {
      setErrorMessage("Please select an FIR image first.")
      return
    }

    setIsLoading(true)
    setErrorMessage(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append("image", selectedFile)

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })

      const payload = (await response.json()) as PredictionResult & { details?: string }

      if (!response.ok) {
        const fullMessage = payload?.details ? `${payload.error}\n${payload.details}` : payload?.error
        throw new Error(fullMessage || "Prediction request failed.")
      }

      setResult(payload)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Something went wrong while predicting."
      setErrorMessage(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_15%_20%,#1f2937_0%,#0f172a_45%,#020617_100%)] px-4 py-8 text-white sm:px-8">
      <div className="mx-auto grid w-full max-w-6xl gap-6 md:grid-cols-[1.1fr_0.9fr]">
        <section className="rounded-3xl border border-white/20 bg-white/10 p-6 shadow-2xl backdrop-blur-xl sm:p-8">
          <p className="mb-3 inline-flex rounded-full border border-cyan-300/40 bg-cyan-300/10 px-3 py-1 text-xs font-semibold tracking-[0.2em] text-cyan-200">
            FORENSIC IPC MAPPER
          </p>
          <h1 className="text-3xl font-extrabold leading-tight sm:text-4xl">
            Upload FIR Image and Test IPC Prediction
          </h1>
          <p className="mt-3 text-sm text-slate-200 sm:text-base">
            This tool checks your FIR image and gives the most likely IPC section in simple words, along with how
            sure the model is.
          </p>

          <form className="mt-6 space-y-4" onSubmit={onSubmit}>
            <label className="block rounded-2xl border border-dashed border-cyan-300/40 bg-slate-900/40 p-5 transition hover:border-cyan-200">
              <span className="mb-3 block text-sm font-medium text-cyan-100">Choose FIR image (PNG/JPG/JPEG)</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={onFileChange}
                className="block w-full cursor-pointer text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:font-semibold file:text-slate-950 hover:file:bg-cyan-400"
              />
            </label>

            <button
              type="submit"
              disabled={!canSubmit}
              className="w-full rounded-xl bg-linear-to-r from-cyan-300 to-emerald-300 px-4 py-3 text-sm font-bold text-slate-950 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? "Running prediction..." : "Run IPC Prediction"}
            </button>
          </form>

          {errorMessage ? (
            <div className="mt-4 rounded-xl border border-rose-300/40 bg-rose-500/20 px-4 py-3 text-sm text-rose-100">
              {errorMessage}
            </div>
          ) : null}

          <div className="mt-5 grid gap-3 text-xs text-slate-300 sm:grid-cols-3 sm:text-sm">
            <div className="rounded-xl border border-white/15 bg-slate-950/40 p-3">Model: Random Forest</div>
            <div className="rounded-xl border border-white/15 bg-slate-950/40 p-3">Classes: 10 IPC Sections</div>
            <div className="rounded-xl border border-white/15 bg-slate-950/40 p-3">Pipeline: IPCV + Features + ML</div>
          </div>
        </section>

        <section className="rounded-3xl border border-white/20 bg-white/10 p-6 shadow-2xl backdrop-blur-xl sm:p-8">
          <h2 className="text-xl font-bold">Preview</h2>

          <div className="mt-4 flex min-h-65 items-center justify-center overflow-hidden rounded-2xl border border-white/15 bg-slate-950/60 p-3">
            {result?.refined_image_base64 ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={`data:${result.refined_image_mime || "image/jpeg"};base64,${result.refined_image_base64}`}
                alt="Refined FIR preview"
                className="max-h-90 w-auto rounded-lg object-contain"
              />
            ) : previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Selected FIR preview" className="max-h-90 w-auto rounded-lg object-contain" />
            ) : (
              <p className="text-center text-sm text-slate-300">Select an FIR image to preview it here.</p>
            )}
          </div>

          <div className="mt-6 rounded-2xl border border-white/15 bg-slate-950/60 p-4">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-cyan-200">Result In Simple Words</h3>

            {!result ? (
              <p className="mt-3 text-sm text-slate-300">No prediction yet. Upload an image and click Run IPC Prediction.</p>
            ) : (
              <div className="mt-3 space-y-3 text-sm">
                <p className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${getTrustPill(result).className}`}>
                  {getTrustPill(result).label}
                </p>
                <p>
                  <span className="text-slate-300">Likely legal category:</span>{" "}
                  <strong className="text-emerald-300">{getPlainCategory(result.ipc_description)}</strong>
                </p>
                <p className="text-slate-200">Legal meaning: {result.ipc_description}</p>
                <p>
                  <span className="text-slate-300">How sure is the model:</span>{" "}
                  <strong className="text-cyan-200">{result.confidence_percentage}</strong>
                  <span className="ml-2 text-xs text-slate-300">({getCertaintyLabel(result.confidence)})</span>
                </p>
                {result.image_quality?.used_variant ? (
                  <p className="text-xs text-slate-300">
                    Refinement used for final verdict: <strong className="text-cyan-100">{result.image_quality.used_variant}</strong> image variant
                  </p>
                ) : null}
                <p className="rounded-lg border border-cyan-300/25 bg-cyan-300/10 px-3 py-2 text-xs text-cyan-100">
                  Refinement summary: {getUserGuidance(result)}
                </p>
                {result.user_recommendation ? (
                  <p className="rounded-lg border border-rose-300/35 bg-rose-500/15 px-3 py-2 text-xs text-rose-100">
                    {result.user_recommendation}
                  </p>
                ) : null}
                <p className="rounded-lg border border-amber-300/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-100">
                  This is AI support, not a final legal decision. Please verify with police or legal experts.
                </p>
                <p className="text-xs text-slate-400">Reference code: {result.predicted_ipc}</p>

                {result.top_k_predictions?.length ? (
                  <div>
                    <h4 className="mb-2 mt-3 text-xs font-semibold uppercase tracking-wider text-slate-300">Other Possible IPC Sections</h4>
                    <div className="space-y-2">
                      {result.top_k_predictions.map((item) => (
                        <div key={`${item.rank}-${item.ipc_code}`} className="rounded-lg border border-white/10 bg-white/5 p-2">
                          <p className="font-semibold text-cyan-100">
                            #{item.rank} {item.ipc_code} ({item.percentage} chance)
                          </p>
                          <p className="text-xs text-slate-300">{item.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="rounded-lg border border-white/10 bg-white/5 p-3 text-xs text-slate-300">
                  Technical feature names are hidden to keep this easy to understand for non-technical users.
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}
