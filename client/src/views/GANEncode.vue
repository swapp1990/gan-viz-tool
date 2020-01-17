<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <div class="p-2">
                <div>raw_images</div>
                <div>
                    <img v-for="img in raw_images" v-bind:src="'data:image/jpeg;base64,'+img.data" @click="alignImages(img.fn)"/>
                </div>
                <!-- <button @click="alignImages()"><i class="icon-cog"></i></button> -->
            </div>
            <div class="p-2">
                <div>aligned_images</div>
                <img v-for="img in aligned_images" v-bind:src="'data:image/jpeg;base64,'+img.data" @click="encodeImages(img.fn)"/>
            </div>
            <div class="p-2">
                <div>encoded_images</div>
                <img v-for="img in encoded_images" v-bind:src="'data:image/jpeg;base64,'+img.data" @click="onEncodeImgClick(img.fn)"/>
            </div>
            <div class="p-2">
                <div>Init</div>
                <hr>
                <button @click="beginTraining()"><i class="icon-bolt"></i></button>
                <div class="p-1">
                    <div v-for="l in logs">{{l.log}}</div>
                </div>
                <hr>
            </div>
            <div v-if="initForPlay" class="p-2">
                <div>Play With Latents</div>
                <hr>
                <div>
                    <input type="range" id="customRange1" min="0" max="1" step="0.1"
                    v-on:change="changeWeights()" v-model="latentW0">
                    <span class="ml-3">Weights: {{latentW0}},{{(1-latentW0)}}</span>
                </div>
                <button @click="playLatents()"><i class="icon-youtube-play"></i></button>
            </div>

            <div class="p-2">
                <img v-for="i in generatedImgs" v-bind:src="'data:image/jpeg;base64,'+i" />
            </div>
            <div class="p-2">
                <img v-for="i in trainingImgs" v-bind:src="'data:image/jpeg;base64,'+i" />
            </div>
            
            <!-- Encoding Loss -->
            <div v-if="gotGraph" class="p-2">
                <span>Loss Graph</span>
                <div class="mlp_div" id="mlp_loss_graph"></div>
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
            generatedImgs: [],
            trainingImgs: [],
            //Latent
            raw_images: [],
            aligned_images: [],
            encoded_images: [],
            initForPlay: false,
            gotGraph: false,
            latentW0: 0.7
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
                else if(content.action == "sendImg") {
                    this.handleReceivedImg(content);
                } else if(content.action == "resetReceivedImgs") {
                    this.resetReceivedImgs(content);
                }
                else if(content.action == "sendRandomGeneratedFigs") {
                    this.showRandomGeneratedFigs(content.fig);
                } else if(content.action == "sendCurrTrainingFigs") {
                    this.showCurrentTrainingFig(content.fig);
                } else if(content.action == "sendGraph") {
                    this.showGraph(content.fig);
                } else if(content.action == "initForPlay") {
                    this.initForPlay = content.val;
                } else if(content.action == "gotGraph") {
                    this.gotGraph = content.val;
                }
            }
        },
        handleReceivedImg(content) {
            if(content.tag == "raw_images") {
                content.fig.axes.forEach(a => {
                    this.raw_images.push({data: a.images[0].data, fn: content.filename});
                });
            } else if(content.tag == "align_images") {
                content.fig.axes.forEach(a => {
                    this.aligned_images.push({data: a.images[0].data, fn: content.filename});
                });
            } else if(content.tag == "encoded_images") {
                content.fig.axes.forEach(a => {
                    let foundImg = this.encoded_images.find(e => e.fn == content.filename)
                    if(foundImg) {
                        foundImg.data = a.images[0].data
                    } else {
                        this.encoded_images.push({data: a.images[0].data, fn: content.filename});
                    }
                });
            }
        },
        resetReceivedImgs(content) {
            if(content.tag == "raw_images") {
                this.raw_images = [];
            } else if(content.tag == "align_images") {
                this.aligned_images = [];
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
        alignImages(fn) {
            this.socket.emit('alignImages', fn);
        },
        encodeImages(fn) {
            this.socket.emit('encodeImages', fn);
        },
        onEncodeImgClick(fn) {

        },
        playLatents() {
            this.socket.emit('playWithLatents', this.latentW0);
        },
        changeWeights() {
            console.log(this.latentW0);
            this.socket.emit('playWithLatents', this.latentW0);
        },
        // showModels(models) {
        //     models.forEach(m => {
        //         let modelObj = JSON.parse(m);
        //         console.log(modelObj);
        //         this.models.push(modelObj);
        //     });
        // },
        showRandomGeneratedFigs(fig) {
            this.generatedImgs = [];
            fig.axes.forEach(a => {
                this.generatedImgs.push(a.images[0].data);
            });
        },
        showCurrentTrainingFig(fig) {
            this.trainingImgs = [];
            fig.axes.forEach(a => {
                this.trainingImgs.push(a.images[0].data);
            });
        },
        showGraph(imgData) {
            let mlpId = '#mlp_loss_graph';
            var graph1 = $(mlpId);
            graph1.html(imgData);
        },
    }
}
</script>
<style lang="stylus" scoped>

</style>