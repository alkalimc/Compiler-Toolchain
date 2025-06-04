<template>
  <el-alert
      type="info"
      :closable="false"
      class="status-alert">
    <template #title>
      <span class="custom-title">部署进度</span>
    </template>

    <div class="status-scroll-container">
      <div class="status-text-container">
        <div v-for="(line, index) in logs" :key="'deploy-' + index">{{ line }}</div>
        <div v-if="logs.length === 0" class="empty-log">暂无部署日志</div>
      </div>
    </div>
  </el-alert>
</template>

<script>
export default {
  name: 'DeployLogDisplay',
  props: {
    logs: {
      type: Array,
      required: true
    }
  }
}
</script>

<style scoped>
.custom-title {
  font-weight: bold;
}

.status-alert {
  background-color:#f5f5f5;
  border: 1px solid #000;
  border-radius: 8px;
  color: #000000;
}

.status-scroll-container {
  max-height: 200px;
  overflow-y: auto;
}

.status-text-container {
  width: 770px;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: #2e2e2e;
  line-height: 1.5;
}
</style>
