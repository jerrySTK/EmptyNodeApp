import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI, ChatOpenAICallOptions } from "@langchain/openai";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";



export class ReactAgentCustom {
    private tools = [new TavilySearchResults({maxResults: 3})];
    private toolNode:ToolNode;
    private model: any;

    constructor() {
        this.toolNode =  new ToolNode(this.tools);
        this.model = new ChatOpenAI({
            model: 'gpt-4o-mini',
            temperature: 0
        }).bindTools(this.tools);
    }

    
    private shouldContinue = ({messages}: typeof MessagesAnnotation.State) => {
        let lastMessage = messages[messages.length -1];
        
        if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
            return 'tools';
        } 
    
        return '__end__';
    }
    
    private callModel = async (state: typeof MessagesAnnotation.State) => {
        const response = await this.model.invoke(state.messages);
    
        return {messages: [response]};
    }

    public getGraph() {
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