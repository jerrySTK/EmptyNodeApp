import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI, ChatOpenAICallOptions } from "@langchain/openai";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { RagWebClient } from "./rag.js";



export class ReactAgentMultiToolCustom {
    private tools:any[] = [];
    private toolNode:ToolNode = new ToolNode([]);
    private model: any;
    private webSource: RagWebClient;

    constructor() {
        this.webSource = new RagWebClient("https://uach.mx/pregrado/licenciatura-en-ingenieria-aeroespacial/");
        
        this.model = new ChatOpenAI({
            model: 'gpt-4o-mini',
            temperature: 0
        })
    }

    private async bindTools() {
        this.tools = [await this.webSource.getRetrieverAsTool(), new TavilySearchResults({maxResults: 3})]
        this.toolNode =  new ToolNode(this.tools);
        this.model = this.model.bindTools(this.tools)
    }
    
    private shouldContinue = ({messages}: typeof MessagesAnnotation.State) => {
        let lastMessage = messages[messages.length -1];
        
        if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
            console.log(lastMessage.tool_calls);
            return 'tools';
        } 
    
        return '__end__';
    }
    
    private callModel = async (state: typeof MessagesAnnotation.State) => {
        const response = await this.model.invoke(state.messages);
    
        return {messages: [response]};
    }

    public async getGraph() {
        await this.bindTools();
        const flow = new StateGraph(MessagesAnnotation)
                    .addNode("agent",this.callModel)
                    .addEdge("__start__","agent")
                    .addNode("tools",this.toolNode)
                    .addEdge("tools","agent")
                    .addConditionalEdges("agent",this.shouldContinue);
        let graph = flow.compile();

        return graph;
    }
    
    
}