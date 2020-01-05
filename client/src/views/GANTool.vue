<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <div class="p-2">
                <button @click="beginTraining()"><i class="icon-bolt"></i></button>
            </div>
            <div class="p-1">
                <div v-for="l in logs">{{l.log}}</div>
            </div>
        </div>
    </div>
</template>
<script>
import $ from 'jquery'
export default {
    name: "GANTool",
    components: {
    },
    computed: {

    },
    data() {
        return {
            //socket
            connected: false,
            socket: null,
            logs: [],
            training: true
        }
    },
    mounted(){
        this.connectSocket();
    },
    methods: {
        reset() {

        },
        getConnectClass() {
            let classes = [];
            if(this.connected) {
                classes.push("btn-primary");
            } else {
                classes.push("btn-danger");
            }
            return classes;
        },
        connectSocket() {
            this.socket = io.connect('http://127.0.0.1:5000');
            this.socket.on('connect',()=>{
                console.log("connected");
                this.onConnected();
            });
            this.socket.on('disconnect',()=>{
                console.log('disconnect');
                this.onDisconnected();
            });
            this.socket.on('connect_error', (error) => {
                console.log("Error");
                this.onDisconnected();
            });
            this.socket.on('error', (err) => {
                console.log("Error!", err);
            });
            this.socket.on('logs',(logs)=>{
                console.log(logs);
                this.handleLogs(logs);
            });
            this.socket.on('General',(content)=>{
                console.log('General ', content.action);
                this.handleGeneralMsg(content);
            });
        },
        onConnected() {
            this.socket.emit('init', this.training);
            this.connected = true;
            this.reset();
        },
        onDisconnected() {
            this.socket.close();
            this.connected = false;
        },
        handleGeneralMsg(content) {
            if(content.action) {
                if(content.action == "sendModelsJson") {
                    this.showModels(content.modelsJson);
                }  
            }
        },
        handleLogs(msg) {
            this.logs.push(msg);
        },
        // GAN TOol
        beginTraining() {
            this.socket.emit('beginTraining');
        },
        showModels(models) {
            models.forEach(m => {
                let modelObj = JSON.parse(m);
                console.log(modelObj);
            });
        }
    }
}
</script>
<style lang="stylus" scoped>

</style>