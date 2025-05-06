<template>
  <div id="app">
    <div class="container">
      <div class="dashboard">
        <DeploymentTool
            @deploy-success="handleDeploySuccess"
            :deployed-models="deployedModels" />
        <DeployedModels
            :models="deployedModels"
            @remove="removeModel" />
      </div>
      <div class="image-section">
        <MyImage />
      </div>
    </div>
  </div>
</template>

<script>
import DeploymentTool from './components/DeploymentTool.vue'
import MyImage from './components/MyImage.vue'
import DeployedModels from './components/DeployedModels.vue'

export default {
  name: 'App',
  components: {
    DeploymentTool,
    MyImage,
    DeployedModels
  },
  data() {
    return {
      deployedModels: []
    }
  },
  methods: {
    handleDeploySuccess(modelData) {
      this.deployedModels.push({
        name: modelData.name,
        quant: modelData.quant,
        time: new Date().toLocaleString()
      })
    },
    removeModel(index) {
      this.deployedModels.splice(index, 1)
      this.$message.success('模型已移除')
    }
  }
}
</script>

<style>
body {
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", Arial, sans-serif;
  margin: 0;
  padding: 20px;
  background-color: #f5f7fa;
}

.container {
  max-width: 80%;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.dashboard {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

/* 控制 DeploymentTool 和 DeployedModels 各占一半 */
.dashboard > * {
  flex: 1 1 0;
}

/*  MyImage 居中 */
.image-section {
  display: flex;
  justify-content: center;
}
</style>