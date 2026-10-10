#!/usr/bin/env python3
"""Bounded Halogen API regression checks; use the full-context vision profile."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import struct
import sys
import urllib.request
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    def call(path, payload=None, stream=False):
        req = urllib.request.Request(args.base_url + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
        with urllib.request.urlopen(req, timeout=600) as response:
            if not stream:
                return json.load(response)
            return [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith('data: ') and line != 'data: [DONE]']

    health = call('/health')
    record = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'health': health, 'checks': []}
    model = health['model']

    def check(name, run):
        try:
            evidence = run()
            record['checks'].append({'name': name, 'passed': True, 'response': evidence})
        except Exception as exc:
            error = {'name': name, 'passed': False, 'error': str(exc)}
            if hasattr(exc, 'read'):
                error['body'] = exc.read().decode()
            record['checks'].append(error)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(name, record['checks'][-1]['passed'], flush=True)

    def chat(messages, **extra):
        return {'model': model, 'messages': messages, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 256, **extra}

    def streaming():
        r = call('/v1/chat/completions', chat([{'role': 'user', 'content': 'Reply with exactly: streaming-ready'}], stream=True), True)
        assert r[0]['choices'][0]['delta']['role'] == 'assistant', r
        assert ''.join(c['choices'][0]['delta'].get('content', '') or '' for c in r if c.get('choices')).strip() == 'streaming-ready', r
        return r
    check('streaming-role-and-content', streaming)

    def tools():
        messages = [{'role': 'user', 'content': 'Use lookup_city to look up Paris. After the result, return only its country.'}]
        definition = [{'type': 'function', 'function': {'name': 'lookup_city', 'description': 'Look up a city', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
        r = call('/v1/chat/completions', chat(messages, tools=definition, stream=True), True)
        assert r[0]['choices'][0]['delta']['role'] == 'assistant', r
        calls = {}
        for chunk in r:
            for choice in chunk.get('choices', []):
                for part in choice['delta'].get('tool_calls', []) or []:
                    entry = calls.setdefault(part['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    if part.get('id'): entry['id'] = part['id']
                    for key in ['name', 'arguments']:
                        entry['function'][key] += part.get('function', {}).get(key, '') or ''
        assert len(calls) == 1, r
        tool = calls[0]
        assert tool['function']['name'] == 'lookup_city' and json.loads(tool['function']['arguments']) == {'city': 'Paris'}, tool
        # Exercise replay of text and calls split into consecutive assistant messages.
        messages += [{'role': 'assistant', 'content': 'I will look up Paris.'}, {'role': 'assistant', 'content': None, 'tool_calls': [tool]}, {'role': 'tool', 'tool_call_id': tool['id'], 'content': '{"country":"France"}'}]
        answer = call('/v1/chat/completions', chat(messages, tools=definition))
        assert answer['choices'][0]['message']['content'].strip() == 'France', answer
        return {'stream': r, 'round_trip': answer}
    check('streaming-tools-and-split-assistant-replay', tools)

    def structured():
        r = call('/v1/chat/completions', chat([{'role': 'user', 'content': 'Return a JSON object with ok set to true.'}], response_format={'type': 'json_schema', 'json_schema': {'name': 'status', 'strict': True, 'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}}}))
        assert json.loads(r['choices'][0]['message']['content']) == {'ok': True}, r
        return r
    check('structured-json', structured)

    def structured_tools_thinking():
        definition = [{'type': 'function', 'function': {'name': 'lookup_city', 'description': 'Look up a city', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
        messages = [
            {'role': 'user', 'content': 'Look up Paris, then return its country in the requested JSON. Once the tool supplies the country, do not call it again.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'city_result', 'type': 'function', 'function': {'name': 'lookup_city', 'arguments': '{"city":"Paris"}'}}]},
            {'role': 'tool', 'tool_call_id': 'city_result', 'content': '{"city":"Paris","country":"France","status":"complete"}'},
        ]
        r = call('/v1/chat/completions', chat(messages, tools=definition,
            reasoning_effort='medium', max_tokens=2048,
            response_format={'type': 'json_schema', 'json_schema': {'name': 'country', 'strict': True, 'schema': {'type': 'object', 'properties': {'country': {'type': 'string'}}, 'required': ['country'], 'additionalProperties': False}}}))
        message = r['choices'][0]['message']
        assert r['usage'].get('completion_tokens_details', {}).get('reasoning_tokens', 0) > 0, r
        assert not message.get('tool_calls'), r
        assert json.loads(message['content']) == {'country': 'France'}, r
        assert r['choices'][0]['finish_reason'] != 'length', r
        return r
    check('structured-tools-with-thinking-after-result', structured_tools_thinking)

    def nullable_arguments(schema):
        definition = [{'type': 'function', 'function': {'name': 'lookup_postcode', 'description': 'Look up the exact postal code, preserving leading zeros.', 'parameters': {'type': 'object', 'properties': {'postcode': schema}, 'required': ['postcode']}}}]
        r = call('/v1/chat/completions', chat(
            [{'role': 'user', 'content': 'Call lookup_postcode with postcode exactly "00123". Preserve the leading zeros.'}],
            tools=definition, tool_choice={'type': 'function', 'function': {'name': 'lookup_postcode'}}))
        calls = r['choices'][0]['message'].get('tool_calls', [])
        assert len(calls) == 1 and calls[0]['function']['name'] == 'lookup_postcode', r
        assert json.loads(calls[0]['function']['arguments']) == {'postcode': '00123'}, r
        return r
    for name, schema in [
        ('type-list', {'type': ['string', 'null']}),
        ('anyof', {'anyOf': [{'type': 'string'}, {'type': 'null'}]}),
        ('oneof', {'oneOf': [{'type': 'string'}, {'type': 'null'}]}),
    ]:
        check('nullable-string-tool-argument-' + name, lambda schema=schema: nullable_arguments(schema))

    def late_instruction(role, between_tool_results=False):
        messages = [{'role': 'user', 'content': 'Say hello.'}, {'role': 'assistant', 'content': 'Hello.'}]
        if between_tool_results:
            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'late_city', 'type': 'function', 'function': {'name': 'lookup_city', 'arguments': '{"city":"Paris"}'}}]})
        messages.append({'role': role, 'content': 'Continue answering concisely.'})
        if between_tool_results:
            messages.append({'role': 'tool', 'tool_call_id': 'late_city', 'content': '{"country":"France"}'})
        messages.append({'role': 'user', 'content': 'Reply with exactly: late-instruction-ready'})
        r = call('/v1/chat/completions', chat(messages))
        assert r['choices'][0]['message']['content'].strip() == 'late-instruction-ready', r
        return r
    for role in ['system', 'developer']:
        check('late-' + role + '-message', lambda role=role: late_instruction(role))
        check('late-' + role + '-between-tool-and-result', lambda role=role: late_instruction(role, True))

    def multiple_streamed_tools():
        definition = [{'type': 'function', 'function': {'name': 'lookup_city', 'description': 'Look up one city per call.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
        payload = chat([{'role': 'user', 'content': 'Call lookup_city twice in this reply: once for Paris and once for Tokyo. Make both calls now. Do not write any text.'}], tools=definition)
        plain = call('/v1/chat/completions', payload)
        chunks = call('/v1/chat/completions', {**payload, 'stream': True}, True)
        content, streamed_calls = '', {}
        for chunk in chunks:
            for choice in chunk.get('choices', []):
                delta = choice['delta']
                content += delta.get('content') or ''
                for part in delta.get('tool_calls', []) or []:
                    entry = streamed_calls.setdefault(part['index'], {'name': '', 'arguments': ''})
                    for key in entry:
                        entry[key] += part.get('function', {}).get(key) or ''
        message = plain['choices'][0]['message']
        expected = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in message.get('tool_calls', [])]
        actual = [(c['name'], json.loads(c['arguments'])) for _, c in sorted(streamed_calls.items())]
        assert len(actual) == len(expected) == 2, {'plain': plain, 'stream': chunks}
        assert actual == expected, {'plain': plain, 'stream': chunks}
        assert content == (message.get('content') or '') == '', {'plain': plain, 'stream': chunks}
        return {'plain': plain, 'stream': chunks}
    check('multiple-tool-stream-content-parity', multiple_streamed_tools)

    def anthropic():
        r = call('/v1/messages', {'model': model, 'max_tokens': 128, 'temperature': 0, 'thinking': {'type': 'disabled'}, 'messages': [{'role': 'user', 'content': 'Reply with exactly: messages-ready'}]})
        assert ''.join(c.get('text', '') for c in r['content'] if c['type'] == 'text').strip() == 'messages-ready', r
        count = call('/v1/messages/count_tokens', {'model': model, 'messages': [{'role': 'user', 'content': 'Hello'}]})
        assert count['input_tokens'] > 0, count
        return {'message': r, 'count': count}
    check('anthropic-messages-and-token-count', anthropic)

    def responses_image():
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 16, 16, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 16) * 16)) + chunk(b'IEND', b'')
        r = call('/v1/responses', {'model': model, 'temperature': 0, 'reasoning': {'effort': 'none'}, 'max_output_tokens': 128, 'input': [
            {'role': 'user', 'content': 'Inspect the image from the tool and reply with only its predominant color.'},
            {'type': 'function_call', 'call_id': 'image_test', 'name': 'read_image', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'image_test', 'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(png).decode()}]},
        ]})
        answer = ''.join(c.get('text', '') for item in r['output'] for c in item.get('content', []))
        assert answer.strip().lower().rstrip('.') == 'red', r
        return r
    check('responses-image-in-tool-result', responses_image)
    return int(any(not c['passed'] for c in record['checks']))


if __name__ == '__main__':
    sys.exit(main())
