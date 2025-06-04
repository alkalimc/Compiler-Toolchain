<template>
  <div id="app">
    <Login v-if="!isLoggedIn" @login-success="handleLoginSuccess" />
    <div v-else class="container">
      <div class="top-section">
        <div class="deployment-wrapper">
          <DeploymentTool
            ref="deploymentTool"
            @deploy-success="handleDeploySuccess"
            @quant-log="handleQuantLog"
            @eval-log="handleEvalLog"
            @deploy-log="handleDeployLog"
            @compile-log="handleCompileLog"
            :auth-info="authInfo" />
        </div>

        <div class="log-displays">
          <QuantLogDisplay :logs="quantLogs" />
          <EvalLogDisplay :logs="evalLogs" />
          <DeployLogDisplay :logs="deployLogs" />
          <CompileLogDisplay :logs="compileLogs" />
        </div>
      </div>

      <div class="bottom-section">
        <DeployedModels
            :auth-info="authInfo" />
      </div>
    </div>
  </div>
</template>

<script>
import Login from './components/Login.vue'
import DeploymentTool from './components/DeploymentTool.vue'
import DeployedModels from './components/DeployedModels.vue'
import QuantLogDisplay from './components/QuantLogDisplay.vue'
import EvalLogDisplay from './components/EvalLogDisplay.vue'
import DeployLogDisplay from './components/DeployLogDisplay.vue'
import CompileLogDisplay from './components/CompileLogDisplay.vue'

export default {
  name: 'App',
  components: {
    Login,
    DeploymentTool,
    DeployedModels,
    QuantLogDisplay,
    EvalLogDisplay,
    DeployLogDisplay,
    CompileLogDisplay
  },
  data() {
    return {
      isLoggedIn: false,
      authInfo: {},
      quantLogs: [],
      evalLogs: [],
      deployLogs: [],
      compileLogs: []
    }
  },
  methods: {
    handleLoginSuccess(credentials) {
      this.authInfo = credentials
      this.isLoggedIn = true
    },
    
    handleQuantLog(newLogs) {
      console.log('收到量化日志:', newLogs)
      if (newLogs.length === 0) {
        this.quantLogs = []
        return
      }
      const lastLength = this.quantLogs.length
      const overlap = newLogs.slice(0, lastLength).every((line, i) => line === this.quantLogs[i])
      if (overlap) {
        this.quantLogs = [...this.quantLogs, ...newLogs.slice(lastLength)]
      } else {
        this.quantLogs = [...new Set([...this.quantLogs, ...newLogs])]
      }
    },

    handleEvalLog(newLogs) {
      console.log('收到评估日志:', newLogs)
      if (newLogs.length === 0) {
        this.evalLogs = []
        return
      }
      const lastLength = this.evalLogs.length
      const overlap = newLogs.slice(0, lastLength).every((line, i) => line === this.evalLogs[i])
      if (overlap) {
        this.evalLogs = [...this.evalLogs, ...newLogs.slice(lastLength)]
      } else {
        this.evalLogs = [...new Set([...this.evalLogs, ...newLogs])]
      }
    },

    handleDeployLog(newLogs) {
      console.log('收到部署日志:', newLogs)
      if (newLogs.length === 0) {
        this.deployLogs = []
        return
      }
      const lastLength = this.deployLogs.length
      const overlap = newLogs.slice(0, lastLength).every((line, i) => line === this.deployLogs[i])
      if (overlap) {
        this.deployLogs = [...this.deployLogs, ...newLogs.slice(lastLength)]
      } else {
        this.deployLogs = [...new Set([...this.deployLogs, ...newLogs])]
      }
    },

    handleCompileLog(newLogs) {
      const lastLength = this.compileLogs.length
      if (newLogs.length === 0) {
        this.compileLogs = []
        return
      }
      const overlap = newLogs.slice(0, lastLength).every((line, i) => line === this.compileLogs[i])
      if (overlap) {
        this.compileLogs = [...this.compileLogs, ...newLogs.slice(lastLength)]
      } else {
        this.compileLogs = [...new Set([...this.compileLogs, ...newLogs])]
      }
    },

    handleDeploySuccess(modelData) {
      try {
        this.handleQuantLog([]);
        this.handleEvalLog([]);
        this.handleDeployLog([]);
        this.handleCompileLog([]);
      } catch (err) {
        this.$reportError(err, {
          action: 'handle_deploy_success',
          modelData: JSON.stringify(modelData)
        })
        this.$message.error('部署记录保存失败')
      }
    },
    globalErrorHandler(err, vm, info) {
      this.$reportError(err, {
        component: vm?.$options?.name,
        lifecycleHook: info,
        stack: err.stack
      })
      console.error('全局捕获的错误:', err)
    }
  },
  mounted() {
    window._unhandledRejection = (event) => {
      this.$reportError(event.reason, {
        type: 'unhandled_rejection'
      })
    }
    window.addEventListener('unhandledrejection', window._unhandledRejection)
  },
  beforeUnmount() {
    if (window._unhandledRejection) {
      window.removeEventListener('unhandledrejection', window._unhandledRejection)
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
  background-image: url('/background.jpg');
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: center center;
}

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.container {
  flex: 1;
  background: rgba(255, 255, 255, 0.95);
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 0 15px rgba(0,0,0,0.1);
}

.top-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  width: 100%;
}

.top-section > .deployment-wrapper {
  max-width: 700px;
  width: 100%;
  margin: 0 auto;
}

.log-displays {
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
}


.log-displays > * {
  max-width: 100%;
  width: 100%;
}

.bottom-section {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 40px;
}
</style>