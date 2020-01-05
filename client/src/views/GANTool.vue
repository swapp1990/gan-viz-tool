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
            <div class="p-2">
                <img v-for="i in generatedImgs" v-bind:src="'data:image/jpeg;base64,'+i" />
            </div>
            <!-- Big GAN -->
            <div class="p-2">
                <span>Loss Graph</span>
                <div class="mlp_div" id="mlp_loss_graph"></div>
            </div>
            <div class="p-2 row">
                <div v-for="m in models" class="col-sm">
                    <div>{{m.config.name}}</div>
                    <div v-for="l in m.config.layers">
                        <div class="row">
                            <div class="col-sm">
                                {{l.class_name}}
                            </div>
                            <!-- <div class="col-sm">
                                {{getLayerShape(l)}}
                            </div> -->
                        </div>
                    </div>
                </div>
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
            training: true,
            //biggan
            models: [],
            generatedImgs: []
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
                } else if(content.action == "sendFigs") {
                    this.showFig(content.fig);
                } else if(content.action == "sendGraph") {
                    this.showGraph(content.fig);
                }
            }
        },
        handleLogs(msg) {
            if(msg.logid) {
                if(msg.type == "replace") {
                    let found = this.logs.find(l => {
                        return l.logid == msg.logid;
                    });
                    if(found) {
                        found.log = msg.log;
                    } else {
                        this.logs.push(msg);
                    }
                }
            } else {
                this.logs.push(msg);
            }
        },
        // GAN TOol
        beginTraining() {
            this.models = []
            this.socket.emit('beginTraining');
        },
        showModels(models) {
            models.forEach(m => {
                let modelObj = JSON.parse(m);
                console.log(modelObj);
                this.models.push(modelObj);
            });
        },
        showFig(fig) {
            this.generatedImgs = [];
            fig.axes.forEach(a => {
                this.generatedImgs.push(a.images[0].data);
            });
        },
        showGraph(imgData) {
            let mlpId = '#mlp_loss_graph';
            var graph1 = $(mlpId);
            graph1.html(imgData);
        },
        // getLayerShape(l) {
        //     // console.log(l);
        //     let l_shape = "unknown shape"
        //     switch(l.class_name) {
        //         case 'InputLayer':
        //             l_shape = "";
        //             l.config.batch_input_shape.forEach(s => {
        //                 if(s == null) {
        //                     l_shape += "bs";
        //                 } else {
        //                     l_shape += s;
        //                 }
        //                 l_shape += ", "
        //             });
        //             break;
        //         case 'Dense':
        //             l_shape = "bs, " + l.config.units;
        //             break;
        //         case 'Reshape':
        //             l_shape = "bs, ";
        //             l.config.target_shape.forEach(s => {
        //                 l_shape += s;
        //                 l_shape += ", "
        //             });
        //             break;
        //         case 'Conv2D':

        //         default:
        //             console.log(l);
        //             console.log("Keras layer type unknown");
        //     }
        //     return l_shape
        // }
    }
}
</script>
<style lang="stylus" scoped>

</style>