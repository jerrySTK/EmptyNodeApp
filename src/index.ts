import * as dotenv from 'dotenv';
import {ChatOpenAI} from '@langchain/openai';
import { HumanMessage,SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import {stdin,stdout} from 'node:process'
import * as readline from 'readline/promises';

import { ReactAgentCustom } from './agent.js';

import { pull } from "langchain/hub";
import { RagWebClient } from './rag.js';
import { ReactAgentMultiToolCustom } from './agent-multitool.js';
dotenv.config();

let basicUsage = async ()=> {

const model = new ChatOpenAI({model: 'gpt-4o-mini'});

const messages = [
    new SystemMessage("Translate from english to spanish"),
    new HumanMessage("Hello world!")
];

const parser = new StringOutputParser();
const chain = model.pipe(parser);

let result = await chain.invoke(messages);
let result2 = await model.invoke(messages);

console.log(result2);
console.log(result);

let systemTemplae = 'Translate the following into {language}';

let template = ChatPromptTemplate.fromMessages([
    ["system",systemTemplae],
    ["user",'{text}']
]);


let promptValue = await template.invoke({
    language: 'russian',
    text: 'Hello world!'
});

console.log(promptValue);

let chain2 = template.pipe(model).pipe(parser);

let result3 = await chain2.invoke({
    language: 'russian',
    text: "Hello world"
});

console.log(result3);
}

const chatbot = async () => {
const chat = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0
});

let response = await chat.invoke([
    new HumanMessage("Hello how are you?, i'm jerry")
]);
console.log(response);

await chat.invoke([new HumanMessage({ content: "What's my name?" })]);
}

const  messageHistories: Record<string,InMemoryChatMessageHistory> = {};

const initChatWithMemory = async () => {
const model = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0
});

const prompt = ChatPromptTemplate.fromMessages([
    ["system",'Eres un asistente util que recuerda todos los detalles que el usuario comparte contigo'],
    ["placeholder",'{chat_history}'],
    ["human", '{input}']
])

const chain = prompt.pipe(model);

const withMessageHistoryRunnable = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: async (sessionId) => {
        if (messageHistories[sessionId] === undefined) {
            messageHistories[sessionId] = new InMemoryChatMessageHistory();
        }
        return messageHistories[sessionId];
    },
    inputMessagesKey: 'input',
    historyMessagesKey: "chat_history"
});

return withMessageHistoryRunnable;
}

var startChatBotWithMemory = async (sessionId:string) => {
var chatWithMemory = await initChatWithMemory();

const rl = readline.createInterface({input: stdin,output:stdout});

const config = {
    configurable: {
        sessionId:sessionId
    }
}

let entry = '';

while (entry !== 'close') {
    
    entry = await rl.question("Input:");
    console.log("User input:",entry);

    if (entry !== 'close') {
        const response = await chatWithMemory.invoke({
                                                    input: entry
        },config);

        console.log("AI response",response.content);
    }
}

rl.close();
}


var testReactAgent = async () => {
    var agent = new ReactAgentCustom();
    
    var graph = agent.getGraph();
    
    var response = await graph.invoke({
        messages: [
            new HumanMessage("Cual es el nombre del recien electo presidente de Mexico?")
        ]
    });
    
    console.log("AI Message: ", response.messages[response.messages.length - 1].content);
}


var customizingPrompt = async () => {
    const ragPrompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
    ragPrompt.promptMessages.push(new SystemMessage('Answer everything in spanish'))
    
}

//await customizingPrompt();
//await startChatBotWithMemory("test123");
//await chatbot();
//await testReactAgent();


var testReacAgentMultiTool = async () => {
    
    const rl = readline.createInterface({input: stdin,output:stdout});
    var agent = new ReactAgentMultiToolCustom();
    
    var graph = await agent.getGraph();
  
    let entry = '';
    
    while (entry !== 'close') {
        
        entry = await rl.question("Input:");
        console.log("User input:",entry);
    
        if (entry !== 'close') {
            var response = await graph.invoke({
                messages: [
                    new HumanMessage(entry)
                ]
            });
            
            console.log("AI Message: ", response.messages[response.messages.length - 1].content);
        }
    }
    
    rl.close();
}

var testRag = async () => {
  
    
    const rl = readline.createInterface({input: stdin,output:stdout});
  
    const ragWebClient = new RagWebClient('https://uach.mx/pregrado/licenciatura-en-artes-visuales/');

    const chain = await ragWebClient.getChain();

  
    let entry = '';
    
    while (entry !== 'close') {
        
        entry = await rl.question("Input:");
        console.log("User input:",entry);
    
        if (entry !== 'close') {
            if (chain) {
                var response = await chain.invoke(entry);
                console.log(response);
            }
        }
    }
    
    rl.close();
}

await testReacAgentMultiTool();



