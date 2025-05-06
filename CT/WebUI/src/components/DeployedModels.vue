<template>
  <el-card class="deployed-models-card">
    <div slot="header" class="header-with-icon">
      <span>已部署模型</span>
    </div>

    <div v-if="models.length === 0" class="empty-message">
      暂无已部署模型
    </div>

    <div v-else class="model-list">
      <div v-for="(model, index) in models" :key="index" class="model-item">
        <div class="model-info">
          <div class="model-name">{{ model.name }}</div>
          <div class="model-quant">量化方式: {{ model.quant }}</div>
          <div class="model-time">部署时间: {{ model.time }}</div>
        </div>
        <el-button
            size="mini"
            type="danger"
            @click="$emit('remove', index)"
            class="remove-btn">
          移除
        </el-button>
      </div>
    </div>
  </el-card>
</template>

<script>
export default {
  name: 'DeployedModels',
  props: {
    models: {
      type: Array,
      default: () => []
    }
  }
}
</script>

<style scoped>
.deployed-models-card {
  height: fit-content;
  margin-left: 20px;
}

.header-with-icon {
  font-size: 18px;
  font-weight: bold;
}

.model-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #f9f9f9;
  border-radius: 4px;
}

.model-info {
  flex: 1;
}

.model-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.model-quant, .model-time {
  font-size: 12px;
  color: #666;
}

.empty-message {
  color: #999;
  text-align: center;
  padding: 20px;
}
</style>