<template>
    <div>
        <button :class="getConnectClass()" @click="connectSocket()"><i class="icon-magnet"></i></button>
        <div v-if="connected">
            <div class="p-2">
                <hr>
                <div>Init</div>
                <button @click="initApp()"><i class="icon-bolt"></i></button>
                <div class="p-1">
                    <div v-for="l in logs">{{l.log}}</div>
                </div>
                <hr>
            </div>
            <div v-if="!alreadyEncoded">
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
                </div>
            </div>
            <div v-if="alreadyEncoded">
                <div>Encoding</div>
                <div class="p-2">
                    <div>encoded_images</div>
                    <img v-for="img in encoded_images" v-bind:src="'data:image/jpeg;base64,'+img.data" @click="onEncodeImgClick(img)"/>
                </div>
                <div class="p-2">
                    <div>selected_images</div>
                    <img v-for="img in selected_images" v-bind:src="'data:image/jpeg;base64,'+img.data"/>
                    <button @click="clearSelected()">Clear</button>
                </div>
                <hr>
            </div>
            <div class="p-2">
                <div>Attributes</div>
                <div>
                    <div v-for="a in attrNames">
                        <input  type="radio" :value="a.type" v-model="selectedAttr">
                        <label>{{a.type}}</label>
                        <div v-for="f in a.factors" class="pl-2">
                            <input  type="radio" :value="f" v-model="selectedFactor">
                            <label class="pl-1">{{f}}</label>
                        </div>
                    </div>
                    <span>Picked: {{ selectedAttr }}: {{selectedFactor}}</span>
                </div>
                <div>
                    <button @click="loadAttributes()"><i class="icon-bolt"></i></button>
                </div>
                <div v-if="dirVecFound">
                    <button @click="generateImgWithDir()"><i class="icon-instagram"></i></button>
                    <input type="range" id="customRange1" min="-5" max="5" step="0.5"
                    v-on:change="changeCoeff()" v-model="dirVecCoeff">
                    <span class="ml-3">Coeff: {{dirVecCoeff}}</span>
                </div>
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
            selected_images: [],
            initForPlay: false,
            dirVecFound: true,
            gotGraph: false,
            alreadyEncoded: true,
            latentW0: 0.7,
            dirVecCoeff: -2.5,
            attrNames: [
                {'type': 'age', 'factors': ['<15', '15-25', '25-35']}, 
                {'type': 'gender', 'factors': ['male', 'female']}
            ],
            selectedAttr: 'gender',
            selectedFactor: 'male'
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
        initApp() {
            let msg = {"alreadyEncoded": this.alreadyEncoded};
            this.socket.emit('initApp', msg);
        },
        alignImages(fn) {
            this.socket.emit('alignImages', fn);
        },
        encodeImages(fn) {
            this.socket.emit('encodeImages', fn);
        },
        loadAttributes() {
            let msg = {'type': this.selectedAttr, 'factor': this.selectedFactor};
            this.socket.emit('loadAttributes', msg);
        },
        generateImgWithDir() {
            let imgNames = this.selected_images.map(si => si.fn);
            let msg = {"images": imgNames, "coeff": this.dirVecCoeff}
            this.socket.emit('generateImgWithDir', msg);
        },
        changeCoeff() {
            let imgNames = this.selected_images.map(si => si.fn);
            let msg = {"images": imgNames, "coeff": this.dirVecCoeff}
            this.socket.emit('generateImgWithDir', msg);
        },
        onEncodeImgClick(img) {
            if(this.selected_images.length < 2) {
                this.selected_images.push(img);
            }
            if(this.selected_images.length == 2) {
                this.initForPlay = true;
            }
        },
        clearSelected() {
            this.selected_images = [];
        },
        playLatents() {
            let imgNames = this.selected_images.map(si => si.fn);
            
            let msg = {"images": imgNames, "latentWs": this.latentW0}
            this.socket.emit('playWithLatents',msg);
        },
        changeWeights() {
            let imgNames = this.selected_images.map(si => si.fn);
            
            let msg = {"images": imgNames, "latentWs": this.latentW0}
            this.socket.emit('playWithLatents', msg);
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