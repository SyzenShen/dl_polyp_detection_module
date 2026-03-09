<template>
  <div class="polyp-detect-container">
    <div class="container">
      <div class="row">
        <div class="col-md-12">
          <h2 class="page-header">Colon Polyp Detection</h2>
          <p class="text-muted">Upload a colonoscopy image to detect polyps automatically.</p>
        </div>
      </div>

      <div class="row polyp-layout">
        <div class="col-md-4">
          <div class="panel panel-default">
            <div class="panel-heading">
              <h3 class="panel-title">Upload Image</h3>
            </div>
            <div class="panel-body">
              <div class="form-group">
                <label>Select Image File</label>
                <input type="file" class="form-control" accept="image/*" @change="onFileSelected" />
              </div>
              <button 
                class="btn btn-primary btn-block" 
                @click="uploadAndDetect" 
                :disabled="!selectedFile || loading"
              >
                <span v-if="loading"><i class="fa fa-spinner fa-spin"></i> Processing...</span>
                <span v-else>Upload & Detect</span>
              </button>
              
              <div v-if="error" class="alert alert-danger mt-3" style="margin-top: 15px;">
                {{ error }}
              </div>
            </div>
          </div>
          
          <div class="panel panel-info" v-if="detections.length > 0">
             <div class="panel-heading">
              <h3 class="panel-title">Detections</h3>
            </div>
            <ul class="detections-inline">
              <li class="det-chip" v-for="(det, index) in detections" :key="index">
                <span class="det-badge" :class="confidenceClass(det.confidence)">
                  {{ (det.confidence * 100).toFixed(1) }}%
                </span>
                <span class="det-label">{{ det.label }}</span>
              </li>
            </ul>
            <div class="panel-footer" v-if="detections.length > 0">
               <button class="btn btn-success btn-block" @click="generateReport" :disabled="explanationLoading">
                 <i class="fa fa-file-text-o"></i> 
                 {{ explanationLoading ? 'Generating Report...' : 'Generate AI Report (Phi-3)' }}
               </button>
            </div>
          </div>
          
          <div class="panel panel-success" v-if="explanation">
            <div class="panel-heading">
              <h3 class="panel-title">Medical Engineering Report</h3>
            </div>
            <div class="panel-body">
              <div class="markdown-body" v-html="renderedExplanation"></div>
            </div>
          </div>

          <div class="alert alert-warning" v-if="resultImage && detections.length === 0 && !loading">
            No polyps detected (Try a clearer image or one with visible polyps).
          </div>
        </div>

        <div v-if="resultImage" class="col-md-8 sticky-visual">
          <div class="panel panel-default">
            <div class="panel-heading">
              <h3 class="panel-title">Visualization</h3>
            </div>
            <div class="panel-body visual-body">
              <div class="image-wrapper">
                <img :src="resultImage" ref="imageRef" @load="onImageLoad" class="visual-img" />
                <canvas ref="canvasRef" class="overlay-canvas"></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, nextTick, computed, watch } from 'vue'
import { useFilesStore } from '../stores/files'
import axios from 'axios'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

export default {
  name: 'PolypDetect',
  setup() {
    const filesStore = useFilesStore()
    const selectedFile = ref(null)
    const loading = ref(false)
    const error = ref(null)
    const resultImage = ref(null)
    const detections = ref([])
    const explanation = ref(null)
    const currentTaskId = ref(null)
    const explanationLoading = ref(false)
    const imageRef = ref(null)
    const canvasRef = ref(null)
    const confidenceClass = (score) => {
      if (score >= 0.8) return 'det-high'
      return 'det-low'
    }

    const onFileSelected = (event) => {
      if (event.target.files.length > 0) {
        selectedFile.value = event.target.files[0]
        // Reset results
        resultImage.value = null
        detections.value = []
        explanation.value = null
        currentTaskId.value = null
        error.value = null
      }
    }

    const uploadAndDetect = async () => {
      if (!selectedFile.value) return
      loading.value = true
      error.value = null
      detections.value = []
      explanation.value = null

      try {
        // 1. Upload File
        const uploadResult = await filesStore.uploadFile(selectedFile.value)
        if (!uploadResult.success) throw new Error(uploadResult.message || 'Upload failed')

        // 2. Find the file ID (Assuming it's the latest one)
        // Refreshing is handled by uploadFile usually, but let's be safe
        // filesStore.files is reactive.
        const latestFile = filesStore.files[0] 
        if (!latestFile) throw new Error('File upload verification failed')

        // 3. Trigger Detection
        const response = await axios.post('/api/ml/trigger/', {
          task_type: 'polyp_detect',
          file_id: latestFile.id
        })
        
        const taskData = response.data
        if (taskData.status === 'failed') {
          throw new Error(taskData.result?.error || 'Detection task failed')
        }

        currentTaskId.value = taskData.task_id
        detections.value = taskData.result?.detections || []
        
        // Show Image
        resultImage.value = latestFile.file // This is the URL from the API response

      } catch (e) {
        console.error(e)
        error.value = e.message || 'An error occurred'
      } finally {
        loading.value = false
      }
    }

    const generateReport = async () => {
      if (!currentTaskId.value) return
      explanationLoading.value = true
      
      try {
        const response = await axios.post(`/api/ml/${currentTaskId.value}/explain/`)
        explanation.value = response.data.explanation
      } catch (e) {
        console.error(e)
        alert('Failed to generate report: ' + (e.response?.data?.detail || e.message))
      } finally {
        explanationLoading.value = false
      }
    }

    const onImageLoad = () => {
      drawBoxes()
    }

    const drawBoxes = () => {
      const img = imageRef.value
      const canvas = canvasRef.value
      if (!img || !canvas) return

      // Set canvas size to match image (rendered size)
      canvas.width = img.width
      canvas.height = img.height

      // Calculate scale factors (Rendered / Natural)
      const scaleX = img.width / img.naturalWidth
      const scaleY = img.height / img.naturalHeight

      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // Box style
      ctx.lineWidth = 3
      ctx.font = '16px Arial'

      detections.value.forEach(det => {
        // Original coordinates
        const [x1, y1, x2, y2] = det.bbox
        
        // Scaled coordinates
        const sx1 = x1 * scaleX
        const sy1 = y1 * scaleY
        const sx2 = x2 * scaleX
        const sy2 = y2 * scaleY
        const width = sx2 - sx1
        const height = sy2 - sy1

        const conf = det.confidence
        const label = det.label

        // Draw Rect
        ctx.strokeStyle = '#00FF00'
        ctx.strokeRect(sx1, sy1, width, height)

        // Draw Label Background
        const text = `${label} ${(conf * 100).toFixed(0)}%`
        const textWidth = ctx.measureText(text).width
        ctx.fillStyle = '#00FF00'
        ctx.fillRect(sx1, sy1 - 20, textWidth + 10, 20)

        // Draw Text
        ctx.fillStyle = '#000000'
        ctx.fillText(text, sx1 + 5, sy1 - 5)
      })
    }

    watch(explanation, () => {
      nextTick(() => {
        drawBoxes()
      })
    })

    return {
      selectedFile,
      loading,
      error,
      resultImage,
      detections,
      imageRef,
      canvasRef,
      onFileSelected,
      uploadAndDetect,
      onImageLoad,
      generateReport,
      currentTaskId,
      explanation,
      explanationLoading,
      confidenceClass,
      renderedExplanation: computed(() => {
        if (!explanation.value) return ''
        const html = marked.parse(explanation.value)
        return DOMPurify.sanitize(html)
      })
    }
  }
}
</script>

<style scoped>
.image-wrapper {
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  position: relative;
  display: inline-block;
  width: 520px;
}
.overlay-canvas {
  z-index: 10;
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}
.markdown-body :deep(h1), .markdown-body :deep(h2), .markdown-body :deep(h3) {
  margin-top: 10px;
  margin-bottom: 10px;
  font-weight: 600;
  line-height: 1.25;
}
.markdown-body :deep(p) {
  margin-bottom: 10px;
  line-height: 1.5;
}
.markdown-body :deep(ul), .markdown-body :deep(ol) {
  padding-left: 20px;
  margin-bottom: 10px;
}
.markdown-body :deep(strong) {
  font-weight: 600;
}
.sticky-visual {
  position: sticky;
  top: 72px; /* 避免被导航栏遮挡 */
  align-self: flex-start;
}
.polyp-layout {
  display: flex;
  align-items: flex-start;
}
.visual-body {
  position: relative;
  display: flex;
  align-items: flex-start;
  justify-content: flex-end; /* 图片靠右 */
  background: #f9f9f9;
  padding: 12px;
  min-height: 400px;
}
.visual-img {
  width: 100%;
  max-width: 100%;
  max-height: 600px;
  display: block;
}
/* 横向显示检测结果为 Chips */
.detections-inline {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  list-style: none;
  margin: 0;
  padding: 10px 12px;
}
.det-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid rgba(27,44,72,0.12);
  border-radius: 16px;
  background: #f8fafc;
  font-size: 14px;
}
.det-badge {
  color: #fff;
  border-radius: 12px;
  padding: 2px 6px;
  font-weight: 600;
}
.det-label {
  color: #333;
  font-weight: 500;
}
.det-badge.det-high {
  background: #eab308; /* 高置信度：黄色 */
  color: #111827;
}
.det-badge.det-low {
  background: #9ca3af; /* 中/低置信度：灰色 */
}
</style>
