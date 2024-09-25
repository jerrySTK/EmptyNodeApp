import * as dotenv from 'dotenv';
import {ChatOpenAI} from '@langchain/openai'

import { HumanMessage,SystemMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';

dotenv.config();

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