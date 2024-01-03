import { HF_ACCESS_TOKEN, HF_TOKEN } from "$env/static/private";
import { buildPrompt } from "$lib/buildPrompt";
import type { TextGenerationStreamOutput } from "@huggingface/inference";
import type { Endpoint } from "../endpoints";
import { z } from "zod";

export const endpointTritonParametersSchema = z.object({
    weight: z.number().int().positive().default(1),
    model: z.any(),
    type: z.literal("triton"),
    maxTokens: z.number().int().positive().default(200),
    badWords: z.array(z.string()).default([]),
    stopWords: z.array(z.string()).default([]),
    url: z.string().url().default("http://localhost:31080/v2/models/ensemble/generate"),
});

export function endpointTirton(
    input: z.input<typeof endpointTritonParametersSchema>
): Endpoint {
    const { url, maxTokens, badWords, stopWords, model } = endpointTritonParametersSchema.parse(input);
    return async ({ conversation }) => {
		const prompt = await buildPrompt({
			messages: conversation.messages,
			webSearch: conversation.messages[conversation.messages.length - 1].webSearch,
			preprompt: conversation.preprompt,
			model,
		});
        console.log("prompt", prompt);
        const requestBody = {
            text_input: prompt,
            max_tokens: 2048,
            stream: true,
            // bad_words: badWords,
            stop_words: "</s>"
        };

        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error(`Failed to generate text: ${await response.text()}`);
        }

        const reader = response.body?.pipeThrough(new TextDecoderStream()).getReader();

        return (async function* () {
            let stop = false;
            let lastOutputLength = 0;
            let generatedText = "";
            let tokenId = 0;
            while (!stop) {
                const out = (await reader?.read()) ?? { done: false, value: undefined };
                if (out.done) {
                    reader?.cancel();
                    return;
                }

                if (!out.value) {
                    return;
                }

                if (out.value.startsWith("data: ")) {
                    let data = null;
                    try {
                        data = JSON.parse(out.value.slice(6));
                    } catch (e) {
                        return;
                    }
                    if (data.text_output || data.sequence_end) {
                        const newText = data.text_output.slice(lastOutputLength);
                        lastOutputLength = data.text_output.length;
                        generatedText = data.text_output;
                        const output: TextGenerationStreamOutput = {
                            token: {
                                id: tokenId++,
                                text: newText, //data.text_output+" " ?? "",
                                logprob: 0,
                                special: false,
                            },
                            generated_text: data.sequence_end ? generatedText : null,
                            details: null,
                        };
                        if (data.sequence_end) {
                            stop = true;
                            reader?.cancel();
                        }
                        yield output;
                    }
                }
            }
        })();
    };
}

export default endpointTirton;

