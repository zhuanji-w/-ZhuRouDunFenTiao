
class ChatInterface {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.fileInput = document.getElementById('fileInput');
        this.fileList = document.getElementById('fileList');
        this.dropZone = document.getElementById('dropZone');
        this.uploadButton = document.getElementById('uploadButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.refreshBtn = document.getElementById('refreshBtn');
        this.genImageCheck = document.getElementById('genImageCheck');
        
        this.selectedFiles = new Map();
        this.isProcessing = false;
        
        this.initEventListeners();
    }

    initEventListeners() {
        // 发送消息
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // 文件选择
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // 拖放功能
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('dragover');
        });
        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('dragover');
        });
        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('dragover');
            this.handleFileDrop(e);
        });

        // 上传按钮
        this.uploadButton.addEventListener('click', () => this.uploadFiles());

        // 刷新索引
        if (this.refreshBtn) {
            this.refreshBtn.addEventListener('click', async () => {
                try {
                    this.showMessage('正在刷新知识库索引...', 'info');
                    const response = await fetch('/refresh', { method: 'POST' });
                    if (response.ok) {
                        this.showMessage('知识库刷新完成！', 'success');
                    } else {
                        this.showMessage('刷新失败', 'error');
                    }
                } catch (error) {
                    this.showMessage('刷新请求出错', 'error');
                }
            });
        }

        // 自动调整输入框高度
        this.chatInput.addEventListener('input', () => this.autoResizeTextarea());
    }

    autoResizeTextarea() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
    }

    handleFileSelect(event) {
        const files = event.target.files;
        this.addFilesToQueue(files);
    }

    handleFileDrop(event) {
        const files = event.dataTransfer.files;
        this.addFilesToQueue(files);
    }

    addFilesToQueue(files) {
        for (let file of files) {
            if (this.isValidFileType(file)) {
                const fileId = Date.now() + Math.random();
                this.selectedFiles.set(fileId, file);
                this.addFileToList(fileId, file);
            } else {
                this.showMessage('不支持的文件类型: ' + file.type, 'error');
            }
        }
        this.fileInput.value = '';
    }

    isValidFileType(file) {
        const validTypes = [
            'application/pdf',
            'text/plain',
            'text/markdown',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ];
        const ext = file.name.split('.').pop().toLowerCase();
        const validExts = ['pdf', 'txt', 'md', 'doc', 'docx'];
        
        return validTypes.includes(file.type) || validExts.includes(ext);
    }

    addFileToList(fileId, file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${file.name}</span>
            <button class="file-remove" onclick="chatInterface.removeFile('${fileId}')">×</button>
        `;
        this.fileList.appendChild(fileItem);
    }

    removeFile(fileId) {
        this.selectedFiles.delete(fileId);
        this.renderFileList();
    }

    renderFileList() {
        this.fileList.innerHTML = '';
        this.selectedFiles.forEach((file, fileId) => {
            this.addFileToList(fileId, file);
        });
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message && this.selectedFiles.size === 0) return;

        // 添加用户消息
        if (message) {
            this.addMessage(message, 'user');
            this.chatInput.value = '';
            this.autoResizeTextarea();
        }

        // 如果有文件，先提示用户上传
        if (this.selectedFiles.size > 0) {
             this.showMessage('检测到文件尚未上传，正在尝试自动上传...', 'info');
             await this.uploadFiles();
        }

        // 发送文本消息
        if (message) {
            this.showTypingIndicator();
            try {
                await this.fetchAIResponse(message);
            } catch (e) {
                this.hideTypingIndicator();
                this.addMessage(`Error: ${e.message}`, 'bot');
            }
        }
    }

    addMessage(text, sender, extras = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString('zh-CN', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        let contentHtml = `<div class="message-text">${this.escapeHtml(text).replace(/\n/g, '<br>')}</div>`;

        // 处理额外内容（图片、来源）
        if (extras) {
             if (extras.images && extras.images.length > 0) {
                contentHtml += `<div class="generated-images">`;
                extras.images.forEach(url => {
                    contentHtml += `<img src="${url}" alt="Generated Image" onclick="window.open(this.src)">`;
                });
                contentHtml += `</div>`;
            }
            
            if (extras.sources && extras.sources.length > 0) {
                contentHtml += `<div class="sources">参考来源:<br>`;
                extras.sources.forEach(src => {
                    contentHtml += `<div class="source-item">
                        <span>${src.source}</span>
                        <span>得分: ${src.score.toFixed(2)}</span>
                    </div>`;
                });
                contentHtml += `</div>`;
            }
        }

        messageDiv.innerHTML = `
            <div class="avatar ${sender}-avatar">${sender === 'user' ? '你' : 'AI'}</div>
            <div class="message-content">
                ${contentHtml}
                <div class="message-time">${timestamp}</div>
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    async fetchAIResponse(userMessage) {
        const generateImage = this.genImageCheck ? this.genImageCheck.checked : false;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: userMessage,
                    generate_image: generateImage
                })
            });

            this.hideTypingIndicator();

            if (response.ok) {
                const data = await response.json();
                this.addMessage(data.answer, 'bot', {
                    images: data.images,
                    sources: data.sources
                });
            } else {
                this.addMessage("服务器响应错误，请稍后再试。", 'bot');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage(`通信错误: ${error.message}`, 'bot');
        }
    }

    async uploadFiles() {
        if (this.selectedFiles.size === 0) {
            this.showMessage('请先选择要上传的文件', 'warning');
            return;
        }

        this.isProcessing = true;
        this.uploadButton.disabled = true;
        this.uploadButton.textContent = '上传中...';

        try {
            // 将 Map 转为数组进行遍历
            const filesArray = Array.from(this.selectedFiles.entries());
            
            for (let [fileId, file] of filesArray) {
                await this.uploadSingleFile(file);
                this.selectedFiles.delete(fileId);
            }
            
            this.renderFileList();
            this.showMessage(`所有文件上传完成`, 'success');
            
        } catch (error) {
            this.showMessage('文件上传部分失败: ' + error.message, 'error');
        } finally {
            this.isProcessing = false;
            this.uploadButton.disabled = false;
            this.uploadButton.textContent = '开始上传文档';
            // 如果全部成功，selectedFiles 应该为空；如果有失败，剩余的还在里面
            this.renderFileList();
        }
    }

    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`文件 ${file.name} 上传失败`);
        }
        return await response.json();
    }

    showMessage(text, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 8px;
            color: white;
            font-size: 0.9rem;
        `;

        const colors = {
            success: 'var(--success)',
            error: 'var(--error)',
            warning: 'var(--warning)',
            info: 'var(--primary)'
        };

        messageDiv.style.background = colors[type] || colors.info;
        messageDiv.textContent = text;

        this.uploadProgress.innerHTML = '';
        this.uploadProgress.appendChild(messageDiv);

        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 5000);
    }
}

// 初始化聊天界面
const chatInterface = new ChatInterface();

// 使全局可访问（用于文件删除按钮）
window.chatInterface = chatInterface;
