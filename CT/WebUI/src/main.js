import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import axios from 'axios'

const API_BASE_URL = 'http://10.20.108.87:7678'

const reportError = async (error, context = {}) => {
    try {
        await axios.post(`${API_BASE_URL}/api/log_client_error`, {
            message: error.stack || error.toString(),
            component: context.componentName || 'unknown',
            route: window.location.pathname,
            ...context  // 扩展其他上下文信息
        }, {
            headers: {
                'Authorization': 'Basic ' + btoa('admin:yuhaolab.CT')
            }
        })
    } catch (e) {
        console.error('[ErrorReport] 上报失败:', e)  // 静默处理上报错误
    }
}

const app = createApp(App)

// 全局错误处理器
app.config.errorHandler = (err, vm, info) => {
    console.error('全局捕获的Vue错误:', err)

    // 上报错误
    reportError(err, {
        componentName: vm?.$options?.name,
        lifecycleHook: info,
        type: 'vue_error_handler'
    })
}

// 未处理的Promise rejection
window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise错误:', event.reason)
    reportError(event.reason, {
        type: 'unhandled_rejection'
    })
})

app.config.globalProperties.$reportError = reportError

app.use(ElementPlus)
app.mount('#app')